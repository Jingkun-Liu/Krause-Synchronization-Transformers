import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from ptflops import get_model_complexity_info
from torchvision.models.vision_transformer import VisionTransformer
import argparse
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import os

from module import (
    SoftTargetCrossEntropy,
    WarmupScheduler,
    get_mixup_cutmix_transforms,
    is_main_process,
    get_rank,
    get_world_size,
    print_rank0,
    USE_BFLOAT16,
    USE_TORCH_COMPILE,
    USE_CHANNELS_LAST,
    USE_FUSED_OPTIMIZER,
)
from data import set_seed, get_imagenet_loaders


def setup_ddp():
    rank = int(os.environ.get('RANK', 0))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
    
    return rank, local_rank, world_size


def cleanup_ddp():
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def train_epoch(model, dataloader, optimizer, criterion, device, warmup_scheduler=None, sampler=None,
                mixup_alpha=0.8, cutmix_alpha=1.0, num_classes=1000, mixup_prob=0.8, label_smoothing=0.1):
    model.train()
    total_loss = torch.tensor(0.0, device=device)
    correct = torch.tensor(0, device=device)
    total = torch.tensor(0, device=device)
    
    use_amp = USE_BFLOAT16 or torch.cuda.is_available()
    amp_dtype = torch.bfloat16 if USE_BFLOAT16 else torch.float16
    
    if sampler is not None:
        sampler.set_epoch(get_rank())

    mixup_cutmix_transform = get_mixup_cutmix_transforms(
        mixup_alpha=mixup_alpha,
        cutmix_alpha=cutmix_alpha,
        num_classes=num_classes,
        mixup_prob=mixup_prob
    )
    
    for batch_idx, (images, labels) in enumerate(dataloader):

        if USE_CHANNELS_LAST:
            images = images.to(device, memory_format=torch.channels_last, non_blocking=True)
        else:
            images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        original_labels = labels.clone()

        if mixup_cutmix_transform is not None:
            images, target = mixup_cutmix_transform(images, labels)
        else:
            target = F.one_hot(labels, num_classes).float()
            if label_smoothing > 0.0:
                target = target * (1 - label_smoothing) + label_smoothing / num_classes
        
        labels = original_labels
        
        optimizer.zero_grad(set_to_none=True)
    
        model_to_check = model.module if isinstance(model, DDP) else model
        
        with torch.amp.autocast('cuda', dtype=amp_dtype, enabled=use_amp):
            
            output = model(images)
            if isinstance(output, (tuple, list)):
                logits = output[0]
            else:
                logits = output
            
        loss = criterion(logits, target)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        if warmup_scheduler is not None:
            warmup_scheduler.step()
        
        total_loss += loss.detach() * labels.size(0)
        _, predicted = logits.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum()
    
    dist.all_reduce(total_loss)
    dist.all_reduce(correct)
    dist.all_reduce(total)
    
    avg_loss = total_loss.item() / total.item()
    acc = 100. * correct.item() / total.item()
    
    return avg_loss, acc


def evaluate(model, dataloader, criterion, device, num_classes=1000, label_smoothing=0.0):
    model.eval()
    total_loss = torch.tensor(0.0, device=device)
    correct = torch.tensor(0, device=device)
    total = torch.tensor(0, device=device)
    
    use_amp = USE_BFLOAT16 or torch.cuda.is_available()
    amp_dtype = torch.bfloat16 if USE_BFLOAT16 else torch.float16
    
    is_soft_target = isinstance(criterion, SoftTargetCrossEntropy)
    
    with torch.no_grad():
        for images, labels in dataloader:
            if USE_CHANNELS_LAST:
                images = images.to(device, memory_format=torch.channels_last, non_blocking=True)
            else:
                images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            model_to_check = model.module if isinstance(model, DDP) else model
            with torch.amp.autocast('cuda', dtype=amp_dtype, enabled=use_amp):
                output = model(images)
                if isinstance(output, (tuple, list)):
                    logits = output[0]
                else:
                    logits = output
            
            if is_soft_target:
                target = F.one_hot(labels, num_classes).float()
                if label_smoothing > 0.0:
                    target = target * (1 - label_smoothing) + label_smoothing / num_classes
                loss = criterion(logits, target)
            else:
                loss = criterion(logits, labels)
            
            total_loss += loss.detach() * labels.size(0)
            _, predicted = logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum()
    
    dist.all_reduce(total_loss)
    dist.all_reduce(correct)
    dist.all_reduce(total)
    
    avg_loss = total_loss.item() / total.item()
    acc = 100. * correct.item() / total.item()
    
    return avg_loss, acc


def visualize_training(standard_results, save_path='vit_s16_training.png'):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    ax1.plot(standard_results['train_accs'], 'o-', label='Standard ViT-S Train', linewidth=2)
    ax1.plot(standard_results['test_accs'], 's-', label='Standard ViT-S Test', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_title('Training Progress (Standard ViT-S/16)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(standard_results['train_losses'], 'o-', label='Standard ViT-S', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.set_title('Training Loss (Standard ViT-S/16)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle('Standard ViT-S/16: Training Results',
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print_rank0(f"\nVisualization saved to {save_path}")


def run_vit_evaluation(args):

    try:
        rank, local_rank, world_size = setup_ddp()
        
        print_rank0("\n" + "="*70)
        print_rank0("STANDARD ViT-S/16 TRAINING")
        print_rank0("="*70 + "\n")
        
        device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
        print_rank0(f"Using device: {device} (rank={rank}, local_rank={local_rank}, world_size={world_size})\n")
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            if is_main_process():
                print(f"CUDA device count: {torch.cuda.device_count()}")
                print(f"CUDA device name: {torch.cuda.get_device_name(local_rank)}")
                print(f"CUDA memory allocated: {torch.cuda.memory_allocated(local_rank) / 1024**2:.2f} MB")
        
        print_rank0("Loading ImageNet dataset...")
        train_loader, test_loader, trainset, testset, train_sampler, test_sampler = get_imagenet_loaders(
            data_path=args.data_path,
            batch_size=args.batch_size,
        )
        
        print_rank0(f"Train samples: {len(trainset)}")
        print_rank0(f"Test samples: {len(testset)}")
        print_rank0(f"Optimizations enabled:")
        print_rank0(f"  - BFloat16: {USE_BFLOAT16}")
        print_rank0(f"  - channels_last: {USE_CHANNELS_LAST}")
        print_rank0(f"  - Fused optimizer: {USE_FUSED_OPTIMIZER}")
        print_rank0(f"  - torch.compile: {USE_TORCH_COMPILE and hasattr(torch, 'compile')}")
        
        model_config = {
            'img_size': args.img_size,
            'patch_size': args.patch_size,
            'num_layers': args.n_layers,
            'num_heads': args.n_heads,
            'hidden_dim': args.d_model,
            'mlp_dim': args.d_ff,
            'num_classes': 1000,
        }
        
        print_rank0("\nInitializing Standard ViT-S/16...")
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        standard_model = VisionTransformer(
            image_size=model_config['img_size'],
            patch_size=model_config['patch_size'],
            num_layers=model_config['num_layers'],
            num_heads=model_config['num_heads'],
            hidden_dim=model_config['hidden_dim'],
            mlp_dim=model_config['mlp_dim'],
            num_classes=model_config['num_classes']
        )
        
        if USE_CHANNELS_LAST:
            standard_model = standard_model.to(device, memory_format=torch.channels_last)
        else:
            standard_model = standard_model.to(device)
        standard_model = DDP(standard_model, device_ids=[local_rank], output_device=local_rank)
        if USE_TORCH_COMPILE and hasattr(torch, 'compile'):
            print_rank0("  Compiling Standard model...")
            standard_model = torch.compile(standard_model, mode='max-autotune')
        
        if is_main_process():
            input_res = (3, args.img_size, args.img_size)
        
            print_rank0("\n" + "="*70)
            print_rank0("COMPUTING MODEL COMPLEXITY (Standard ViT-S/16)")
            print_rank0("="*70)
            
            standard_params = sum(p.numel() for p in standard_model.module.parameters())
            print_rank0(f"\n[Standard ViT]")
            print_rank0(f"  Parameters: {standard_params:,}")
        
            try:
                flops_standard, _ = get_model_complexity_info(
                    standard_model.module, input_res, as_strings=False,
                    print_per_layer_stat=False
                )
                print_rank0(f"  FLOPs: {int(flops_standard):,} ({flops_standard/1e9:.2f} GFLOPs)")
            except Exception as e:
                print_rank0(f"  FLOPs calculation failed: {e}")
                flops_standard = 0
        
        dist.barrier()

        criterion = SoftTargetCrossEntropy()
        print_rank0(f"Using SoftTargetCrossEntropy loss (label_smoothing={args.label_smoothing})")
        
        optimizer_kwargs = {'lr': args.lr, 'weight_decay': args.weight_decay}
        if USE_FUSED_OPTIMIZER and hasattr(torch.optim.AdamW, '__init__'):
            import inspect
            if 'fused' in inspect.signature(torch.optim.AdamW.__init__).parameters:
                optimizer_kwargs['fused'] = True
                print_rank0("  Using fused AdamW optimizer")
        
        std_optimizer = torch.optim.AdamW(standard_model.parameters(), **optimizer_kwargs)
        
        std_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(std_optimizer, T_max= args.epochs - args.warmup_epochs)
        
        iterations_per_epoch = len(train_loader)
        warmup_steps = args.warmup_epochs * iterations_per_epoch
        
        print_rank0(f"\nWarmup Configuration:")
        print_rank0(f"  Warmup epochs: {args.warmup_epochs}")
        print_rank0(f"  Iterations per epoch: {iterations_per_epoch}")
        print_rank0(f"  Total warmup steps: {warmup_steps}")
        print_rank0(f"  Target learning rate: {args.lr}")
        
        std_warmup_scheduler = WarmupScheduler(std_optimizer, warmup_steps=warmup_steps, start_lr=0.0, target_lr=args.lr) if args.warmup_epochs > 0 else None
        print_rank0("\n" + "="*70)
        print_rank0("Training Standard ViT-S/16 (torchvision)" + (" with warmup" if args.warmup_epochs > 0 else ""))
        print_rank0("="*70)
        
        standard_results = {
            'train_losses': [],
            'train_accs': [],
            'test_losses': [],
            'test_accs': []
        }
        
        best_std_acc = 0
        best_std_epoch = 0
        for epoch in range(args.epochs):
            train_sampler.set_epoch(epoch)
            current_warmup_scheduler = std_warmup_scheduler if (std_warmup_scheduler and std_warmup_scheduler.is_warming_up()) else None
            train_loss, train_acc = train_epoch(
                standard_model, train_loader, std_optimizer, criterion, device,
                warmup_scheduler=current_warmup_scheduler, sampler=train_sampler,
                mixup_alpha=args.mixup_alpha, cutmix_alpha=args.cutmix_alpha,
                num_classes=model_config['num_classes'], mixup_prob=args.mixup_prob,
                label_smoothing=args.label_smoothing
            )
            test_loss, test_acc = evaluate(standard_model, test_loader, criterion, device,
                                           num_classes=model_config['num_classes'],
                                           label_smoothing=args.label_smoothing)
            
            standard_results['train_losses'].append(train_loss)
            standard_results['train_accs'].append(train_acc)
            standard_results['test_losses'].append(test_loss)
            standard_results['test_accs'].append(test_acc)
            
            if std_warmup_scheduler is None or not std_warmup_scheduler.is_warming_up():
                std_scheduler.step()
            
            if test_acc > best_std_acc:
                best_std_acc = test_acc
                best_std_epoch = epoch + 1
                if is_main_process():
                    torch.save(standard_model.module.state_dict(), 'best_standard_vit_ImageNet_vits16.pth')
                    print_rank0(f"  Saved best Standard-ViT-S weights (Epoch {epoch+1}, Test Acc: {test_acc:.2f}%)")
            
            current_lr = std_optimizer.param_groups[0]['lr']
            if (epoch + 1) % 1 == 0 :
                lr_info = f" - LR: {current_lr:.6f}" if args.warmup_epochs > 0 and epoch < args.warmup_epochs else ""
                print_rank0(f"Epoch {epoch+1}/{args.epochs} - Train: {train_acc:.2f}% - Test: {test_acc:.2f}% - Best: {best_std_acc:.2f}%{lr_info}")

        print_rank0("\n" + "="*70)
        print_rank0("LOADING BEST MODEL FOR FINAL EVALUATION")
        print_rank0("="*70)

        print_rank0("\n[Loading best Standard ViT weights]")
        state_dict = torch.load('best_standard_vit_ImageNet_vits16.pth', map_location=device)
        standard_model.module.load_state_dict(state_dict)
        standard_model.eval()
        best_std_loss, best_std_acc = evaluate(standard_model, test_loader, criterion, device,
                                                num_classes=model_config['num_classes'],
                                                label_smoothing=args.label_smoothing)
        print_rank0(f"Standard ViT Best Model (Epoch {best_std_epoch}):")
        print_rank0(f"  Test Loss: {best_std_loss:.4f}, Test Acc: {best_std_acc:.2f}%")
        
        print_rank0("\n" + "="*70)
        print_rank0("EVALUATION SUMMARY")
        print_rank0("="*70)
        
        print_rank0("\n Classification Performance:")
        print_rank0(f"  Standard ViT-S: Best Test Accuracy = {best_std_acc:.2f}% (Epoch {best_std_epoch})")
        
        if is_main_process():
            print_rank0("\n" + "="*70)
            print_rank0("Generating Visualizations...")
            print_rank0("="*70)
            
            visualize_training(standard_results, args.save_path)
            
            print_rank0("\nEvaluation complete!")
            print_rank0(f"\nGenerated files:")
            print_rank0(f"  - {args.save_path}: Training curves (accuracy and loss)")
        
        dist.barrier()
    
    finally:
        cleanup_ddp()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Train Standard ViT-S/16 on ImageNet')
    
    parser.add_argument('--data_path', type=str, default='./data/ImageNet',
                       help='Path to ImageNet dataset (default: ./data/ImageNet)')
    parser.add_argument('--batch_size', type=int, default=256,
                       help='Batch size for training and evaluation (default: 256)')
    
    parser.add_argument('--img_size', type=int, default=224,
                       help='Input image size (default: 224, ImageNet standard)')
    parser.add_argument('--patch_size', type=int, default=16,
                       help='Patch size for ViT (default: 16, ViT-S/16 standard)')
    parser.add_argument('--d_model', type=int, default=384,
                       help='Embedding dimension (default: 384, ViT-S-16)')
    parser.add_argument('--n_heads', type=int, default=6,
                       help='Number of attention heads (default: 6, ViT-S-16)')
    parser.add_argument('--n_layers', type=int, default=12,
                       help='Number of transformer layers (default: 12, ViT-S-16)')
    parser.add_argument('--d_ff', type=int, default=1536,
                       help='Feed-forward network dimension')
    parser.add_argument('--dropout', type=float, default=0.0,
                       help='Dropout rate')
    parser.add_argument('--epochs', type=int, default=300,
                       help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                       help='Weight decay for AdamW optimizer')
    parser.add_argument('--warmup_epochs', type=int, default=10,
                       help='Number of warmup epochs')
    
    parser.add_argument('--drop_path', type=float, default=0.1,
                       help='Drop path rate for Stochastic Depth')
    parser.add_argument('--mixup_alpha', type=float, default=0.8,
                       help='Mixup alpha parameter')
    parser.add_argument('--cutmix_alpha', type=float, default=1.0,
                       help='CutMix alpha parameter')
    parser.add_argument('--mixup_prob', type=float, default=0.8,
                       help='Probability of using CutMix vs Mixup when both enabled')
    parser.add_argument('--label_smoothing', type=float, default=0.1,
                       help='Label smoothing factor')
    
    parser.add_argument('--no_cuda', action='store_true',
                       help='Disable CUDA even if available')
    parser.add_argument('--save_path', type=str, default='vit_imagenet_vits16.png',
                       help='Path to save visualization')

    args = parser.parse_args()

    if int(os.environ.get('RANK', 0)) == 0:
        print("\n" + "="*70)
        print("CONFIGURATION")
        print("="*70)
        print(f"Data path: {args.data_path}")
        print(f"Batch size per GPU: {args.batch_size}")
        print(f"Model: img_size={args.img_size}, patch_size={args.patch_size}, d_model={args.d_model}")
        print(f"Training: epochs={args.epochs}, lr={args.lr}, weight_decay={args.weight_decay}, warmup_epochs={args.warmup_epochs}")
        print(f"DeiT Recipe: drop_path={args.drop_path}, mixup_alpha={args.mixup_alpha}, cutmix_alpha={args.cutmix_alpha}, label_smoothing={args.label_smoothing}")
        print(f"Device: {'cuda' if not args.no_cuda and torch.cuda.is_available() else 'cpu'}")
        print(f"Distributed: Using torchrun (WORLD_SIZE will be set automatically)")
        print(f"Visualization save path: {args.save_path}")
        print("="*70 + "\n")

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    set_seed(3407)

    run_vit_evaluation(args)
