import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import argparse
import warnings
import logging
from fvcore.nn import FlopCountAnalysis
from module import (
    WarmupScheduler,
    SoftTargetCrossEntropy,
    get_mixup_cutmix_transforms,
    create_krause_swin_t,
    get_all_sigma_values
)
from data import set_seed, get_cifar10_loaders


def train_epoch(model, dataloader, optimizer, criterion, device, warmup_scheduler=None,
                mixup_alpha=0.8, cutmix_alpha=1.0, num_classes=10, mixup_prob=0.8, label_smoothing=0.1):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    mixup_cutmix_transform = get_mixup_cutmix_transforms(
        mixup_alpha=mixup_alpha,
        cutmix_alpha=cutmix_alpha,
        num_classes=num_classes,
        mixup_prob=mixup_prob
    )
    
    for batch_idx, (images, labels) in enumerate(dataloader):
        images, labels = images.to(device), labels.to(device)
        
        original_labels = labels.clone()

        if mixup_cutmix_transform is not None:
            images, target = mixup_cutmix_transform(images, labels)
            
            if label_smoothing > 0.0:
                target = target * (1 - label_smoothing) + label_smoothing / num_classes
        else:
            target = F.one_hot(labels, num_classes).float()
            if label_smoothing > 0.0:
                target = target * (1 - label_smoothing) + label_smoothing / num_classes
        
        labels = original_labels
        
        optimizer.zero_grad()
    
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
        
        total_loss += loss.item()
        _, predicted = logits.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    return total_loss / len(dataloader), 100. * correct / total


def evaluate(model, dataloader, criterion, device, num_classes=10, label_smoothing=0.0):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    is_soft_target = isinstance(criterion, SoftTargetCrossEntropy)
    
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            
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
            
            total_loss += loss.item()
            _, predicted = logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    return total_loss / len(dataloader), 100. * correct / total


def visualize_comprehensive_analysis(krause_results, save_path='kswint_cifar10_analysis.png'):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    ax1.plot(krause_results['train_accs'], 'o-', label='Krause-Swin Train', linewidth=2)
    ax1.plot(krause_results['test_accs'], 's-', label='Krause-Swin Test', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_title('Training Progress')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(krause_results['train_losses'], 'o-', label='Krause-Swin', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.set_title('Training Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle('Krause-SwinTransformer Training Results',
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Visualization saved to {save_path}")


def run_kswint_evaluation(args):
    print("\n" + "="*70)
    print("KRAUSE SWIN TRANSFORMER-TINY TRAINING BENCHMARK (CIFAR-10)")
    print("="*70 + "\n")
    
    device = torch.device('cuda' if not args.no_cuda and torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
        print(f"CUDA memory allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
    
    print("Loading CIFAR-10 dataset...")
    train_loader, test_loader, trainset, testset = get_cifar10_loaders(args.data_path, args.batch_size)
    
    print(f"Train samples: {len(trainset)}")
    print(f"Test samples: {len(testset)}")
    
    print("\nInitializing models...")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    krause_swin_model = create_krause_swin_t(num_classes=10, sigma=args.sigma, drop_path=args.drop_path)
    
    print("\n" + "="*70)
    print("COMPUTING MODEL COMPLEXITY")
    print("="*70)
    
    fvcore_logger = logging.getLogger("fvcore")
    old_level = fvcore_logger.level
    fvcore_logger.setLevel(logging.ERROR)
    
    krause_params = sum(p.numel() for p in krause_swin_model.parameters())
    print(f"\n[Krause-SwinTransformer-tiny]")
    print(f"  Parameters: {krause_params:,}")

    try:
        dummy_input = torch.randn(1, 3, 32, 32)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            flop_counter = FlopCountAnalysis(krause_swin_model, (dummy_input,))
            flops_krause = flop_counter.total()
        print(f"  FLOPs: {int(flops_krause):,} ({flops_krause/1e9:.2f} GFLOPs)")
    except Exception as e:
        print(f"  FLOPs calculation failed: {e}")
        flops_krause = 0
    
    fvcore_logger.setLevel(old_level)
    
    print("\nMoving model to device...")
    krause_swin_model = krause_swin_model.to(device)
    
    criterion = SoftTargetCrossEntropy()
    print(f"  Using SoftTargetCrossEntropy loss (label_smoothing={args.label_smoothing})")
    
    iterations_per_epoch = len(train_loader)
    warmup_steps = args.warmup_epochs * iterations_per_epoch
    
    print(f"\nWarmup Configuration:")
    print(f"  Warmup epochs: {args.warmup_epochs}")
    print(f"  Iterations per epoch: {iterations_per_epoch}")
    print(f"  Total warmup steps: {warmup_steps}")
    print(f"  Target learning rate: {args.lr}")

    print("\n" + "="*70)
    print("Training Krause-SwinTransformer-tiny" + (" (with warmup)" if args.warmup_epochs > 0 else ""))
    print("="*70)
    
    krause_optimizer = torch.optim.AdamW(krause_swin_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    krause_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(krause_optimizer, T_max=args.epochs - args.warmup_epochs)
    krause_warmup_scheduler = WarmupScheduler(krause_optimizer, warmup_steps=warmup_steps, start_lr=0.0, target_lr=args.lr) if args.warmup_epochs > 0 else None
    
    krause_results = {
        'train_losses': [],
        'train_accs': [],
        'test_losses': [],
        'test_accs': []
    }
    
    best_krause_acc = 0
    best_krause_epoch = 0
    for epoch in range(args.epochs):
        current_warmup_scheduler = krause_warmup_scheduler if (krause_warmup_scheduler and krause_warmup_scheduler.is_warming_up()) else None
        train_loss, train_acc = train_epoch(
            krause_swin_model, train_loader, krause_optimizer, criterion, device,
            warmup_scheduler=current_warmup_scheduler,
            mixup_alpha=args.mixup_alpha, cutmix_alpha=args.cutmix_alpha,
            num_classes=10, mixup_prob=args.mixup_prob,
            label_smoothing=args.label_smoothing
        )
        test_loss, test_acc = evaluate(krause_swin_model, test_loader, criterion, device,
                                        num_classes=10,
                                        label_smoothing=args.label_smoothing)
        
        krause_results['train_losses'].append(train_loss)
        krause_results['train_accs'].append(train_acc)
        krause_results['test_losses'].append(test_loss)
        krause_results['test_accs'].append(test_acc)
        
        if krause_warmup_scheduler is None or not krause_warmup_scheduler.is_warming_up():
            krause_scheduler.step()
        
        if test_acc > best_krause_acc:
            best_krause_acc = test_acc
            best_krause_epoch = epoch + 1
            torch.save(krause_swin_model.state_dict(), 'best_krause_swin_cifar10.pth')
            print(f"  → Saved best Krause-Swin weights (Epoch {epoch+1}, Test Acc: {test_acc:.2f}%)")
        
        current_lr = krause_optimizer.param_groups[0]['lr']

        sigma_values = get_all_sigma_values(krause_swin_model)
        sigma_info_parts = [f"L{i+1}:{sigma:.4f}" for i, (name, sigma) in enumerate(sigma_values)]
        sigma_info = " - Sigma: " + " | ".join(sigma_info_parts) if sigma_info_parts else ""
        
        if (epoch + 1) % 1 == 0:
            lr_info = f" - LR: {current_lr:.6f}" if args.warmup_epochs > 0 and epoch < args.warmup_epochs else ""
            print(f"Epoch {epoch+1}/{args.epochs} - Train: {train_acc:.2f}% - Test: {test_acc:.2f}% - Best: {best_krause_acc:.2f}%{lr_info}{sigma_info}")
    
    print("\n" + "="*70)
    print("LOADING BEST KRAUSE MODEL FOR FINAL EVALUATION")
    print("="*70)
    
    print("\n[Loading best Krause-SwinTransformer weights]")
    state_dict = torch.load('best_krause_swin_cifar10.pth', map_location=device)
    krause_swin_model.load_state_dict(state_dict)
    krause_swin_model.eval()
    best_krause_loss, best_krause_acc = evaluate(krause_swin_model, test_loader, criterion, device,
                                                    num_classes=10,
                                                    label_smoothing=args.label_smoothing)
    print(f"Krause-SwinTransformer Best Model (Epoch {best_krause_epoch}):")
    print(f"  Test Loss: {best_krause_loss:.4f}, Test Acc: {best_krause_acc:.2f}%")
    
    
    print("\n" + "="*70)
    print("FINAL EVALUATION SUMMARY")
    print("="*70)
    
    print("\n Classification Performance:")
    print(f"  Krause-SwinTransformer: Best Test Accuracy = {best_krause_acc:.2f}% (Epoch {best_krause_epoch})")
    
    print("\n" + "="*70)
    print("Generating Visualizations...")
    print("="*70)
    
    visualize_comprehensive_analysis(krause_results, args.save_path)
    
    print("Evaluation complete!")
    print(f"Generated files:")
    print(f"  - {args.save_path}: Training curves (accuracy and loss)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Krause-SwinTransformer-tiny on CIFAR-10')
    
    parser.add_argument('--data_path', type=str, default='./data',
                       help='Path to store CIFAR-10 dataset (default: ./data)')
    parser.add_argument('--batch_size', type=int, default=256,
                       help='Batch size for training and evaluation (default: 256)')
    
    parser.add_argument('--img_size', type=int, default=32,
                       help='Input image size (default: 32, CIFAR-10 standard)')
    
    parser.add_argument('--epochs', type=int, default=300,
                       help='Number of training epochs (default: 300)')
    parser.add_argument('--lr', type=float, default=2e-3,
                       help='Learning rate (default: 2e-3)')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                       help='Weight decay for AdamW optimizer (default: 0.01)')
    parser.add_argument('--warmup_epochs', type=int, default=20,
                       help='Number of warmup epochs (default: 20)')
    
    parser.add_argument('--drop_path', type=float, default=0.1,
                       help='Drop path rate for Stochastic Depth')
    parser.add_argument('--mixup_alpha', type=float, default=0.8,
                       help='Mixup alpha parameter (default: 0.8)')
    parser.add_argument('--cutmix_alpha', type=float, default=1.0,
                       help='CutMix alpha parameter (default: 1.0)')
    parser.add_argument('--mixup_prob', type=float, default=0.8,
                       help='Probability of using CutMix vs Mixup when both enabled (default: 0.8)')
    parser.add_argument('--label_smoothing', type=float, default=0.1,
                       help='Label smoothing factor (default: 0.1)')
    
    parser.add_argument('--sigma', type=float, default=4.0,
                       help='Initial sigma for Krause/RBF attention (default: 4.0)')
    
    parser.add_argument('--no_cuda', action='store_true',
                       help='Disable CUDA even if available')
    parser.add_argument('--save_path', type=str, default='swin_cifar10_swint16.png',
                       help='Path to save visualization (default: swin_cifar10_swint16.png)')

    args = parser.parse_args()

    set_seed(3407)

    print("\n" + "="*70)
    print("CONFIGURATION")
    print("="*70)
    print(f"Data path: {args.data_path}")
    print(f"Batch size: {args.batch_size}")
    print(f"Model: Krause-SwinTransformer-tiny, img_size={args.img_size}")
    print(f"Training: epochs={args.epochs}, lr={args.lr}, weight_decay={args.weight_decay}, warmup_epochs={args.warmup_epochs}")
    print(f"drop_path={args.drop_path}, mixup_alpha={args.mixup_alpha}, cutmix_alpha={args.cutmix_alpha}, label_smoothing={args.label_smoothing}")
    print(f"Krause Attention: sigma={args.sigma}")
    print(f"Device: {'cuda' if not args.no_cuda and torch.cuda.is_available() else 'cpu'}")
    print(f"Visualization save path: {args.save_path}")
    print("="*70 + "\n")

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    run_kswint_evaluation(args)
