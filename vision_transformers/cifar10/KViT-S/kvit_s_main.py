import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import argparse
from ptflops import get_model_complexity_info
from module import (
    KrauseVisionTransformer, 
    KrauseViTAttention,
    SoftTargetCrossEntropy,
    WarmupScheduler,
    get_mixup_cutmix_transforms,
    count_krause_attention_flops
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
    
        logits, _, _ = model(images)
        
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
            
            logits, _, _ = model(images)

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


def visualize_comprehensive_analysis(results, save_path='vit_analysis.png'):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1.plot(results['train_accs'], 'o-', label='Krause-ViT Train', linewidth=2)
    ax1.plot(results['test_accs'], 's-', label='Krause-ViT Test', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_title('Training Progress')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(results['train_losses'], 'o-', label='Krause-ViT', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.set_title('Training Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle('Krause-ViT (RBF/Top-K): Training Curves',
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n Visualization saved to {save_path}")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Krause-ViT on CIFAR-10')
    
    parser.add_argument('--data_path', type=str, default='./data',
                       help='Path to store CIFAR-10 dataset (default: ./data)')
    parser.add_argument('--batch_size', type=int, default=256,
                       help='Batch size for training and evaluation (default: 256)')
    
    parser.add_argument('--img_size', type=int, default=32,
                       help='Input image size (default: 32)')
    parser.add_argument('--patch_size', type=int, default=4,
                       help='Patch size for ViT (default: 4)')
    parser.add_argument('--d_model', type=int, default=384,
                       help='Embedding dimension (default: 384, ViT-S)')
    parser.add_argument('--n_heads', type=int, default=6,
                       help='Number of attention heads (default: 6, ViT-S)')
    parser.add_argument('--n_layers', type=int, default=12,
                       help='Number of transformer layers (default: 12, ViT-S)')
    parser.add_argument('--d_ff', type=int, default=1536,
                       help='Feed-forward network dimension (default: 1536, 384*4)')
    parser.add_argument('--dropout', type=float, default=0.0,
                       help='Dropout rate (default: 0.0)')
    parser.add_argument('--sigma', type=float, default=2.5,
                       help='Initial sigma for Krause/RBF attention (default: 2.5)')
    parser.add_argument('--top_k', type=int, default=2,
                       help='*Minimum* Top-K for Krause attention (first layer) (default: 2)')
    
    parser.add_argument('--epochs', type=int, default=300,
                       help='Number of training epochs (default: 300)')
    parser.add_argument('--lr', type=float, default=3e-4,
                       help='Learning rate (default: 3e-4)')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                       help='Weight decay for AdamW optimizer (default: 0.05)')
    parser.add_argument('--warmup_epochs', type=int, default=10,
                       help='Number of warmup epochs (default: 10)')
    
    parser.add_argument('--drop_path', type=float, default=0.1,
                       help='Drop path rate for Stochastic Depth (default: 0.1)')
    parser.add_argument('--mixup_alpha', type=float, default=0.8,
                       help='Mixup alpha parameter (default: 0.8)')
    parser.add_argument('--cutmix_alpha', type=float, default=1.0,
                       help='CutMix alpha parameter (default: 1.0)')
    parser.add_argument('--mixup_prob', type=float, default=0.8,
                       help='Probability of using CutMix vs Mixup when both enabled (default: 0.8)')
    parser.add_argument('--label_smoothing', type=float, default=0.1,
                       help='Label smoothing factor (default: 0.1)')
    
    parser.add_argument('--no_cuda', action='store_true',
                       help='Disable CUDA even if available')
    parser.add_argument('--save_path', type=str, default='krause_vit_analysis.png',
                       help='Path to save visualization (default: krause_vit_analysis.png)')

    args = parser.parse_args()

    set_seed(3407)

    def run_vit_evaluation(
        data_path=args.data_path,
        batch_size=args.batch_size,
        img_size=args.img_size,
        patch_size=args.patch_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=args.d_ff,
        dropout=args.dropout,
        sigma=args.sigma,
        top_k=args.top_k,
        n_epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        use_cuda=not args.no_cuda,
        save_path=args.save_path,
        warmup_epochs=args.warmup_epochs,
        drop_path=args.drop_path,
        mixup_alpha=args.mixup_alpha,
        cutmix_alpha=args.cutmix_alpha,
        mixup_prob=args.mixup_prob,
        label_smoothing=args.label_smoothing
    ):
        print("\n" + "="*70)
        print("KRAUSE VISION TRANSFORMER EVALUATION")
        print("="*70 + "\n")
        
        device = 'cuda' if use_cuda and torch.cuda.is_available() else 'cpu'
        print(f"Using device: {device}\n")
        
        print("Loading CIFAR-10 dataset...")
        train_loader, test_loader, trainset, testset = get_cifar10_loaders(data_path, batch_size)
        
        print(f" Train samples: {len(trainset)}")
        print(f" Test samples: {len(testset)}")

        model_config = {
            'img_size': img_size,
            'patch_size': patch_size,
            'in_channels': 3,
            'num_classes': 10,
            'd_model': d_model,
            'n_heads': n_heads,
            'n_layers': n_layers,
            'd_ff': d_ff,
            'dropout': dropout
        }

        print("\nInitializing model...")

        model = KrauseVisionTransformer(
            **model_config,
            sigma=sigma,
            top_k=top_k,
            drop_path=drop_path,
        ).to(device)

        input_res = (3, img_size, img_size)

        print("\n" + "="*70)
        print("COMPUTING MODEL COMPLEXITY")
        print("="*70)

        custom_hooks = { KrauseViTAttention: count_krause_attention_flops }
        
        model_params = sum(p.numel() for p in model.parameters())
        print(f"\n[Krause-ViT]")
        print(f"  Parameters: {model_params:,}")
    
        try:
            flops, _ = get_model_complexity_info(
                model, input_res, as_strings=False, 
                print_per_layer_stat=False, custom_modules_hooks=custom_hooks
            )
            print(f"  FLOPs: {int(flops):,} ({flops/1e9:.2f} GFLOPs)")
        except Exception as e:
            print(f"  FLOPs calculation failed: {e}")

        criterion = SoftTargetCrossEntropy()
        print(f"  Using SoftTargetCrossEntropy loss (label_smoothing={label_smoothing})")
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs-warmup_epochs)

        iterations_per_epoch = len(train_loader)
        warmup_steps = warmup_epochs * iterations_per_epoch
        
        print(f"\nWarmup Configuration:")
        print(f"  Warmup epochs: {warmup_epochs}")
        print(f"  Iterations per epoch: {iterations_per_epoch}")
        print(f"  Total warmup steps: {warmup_steps}")
        print(f"  Target learning rate: {lr}")
        
        warmup_scheduler = WarmupScheduler(optimizer, warmup_steps=warmup_steps, start_lr=0.0, target_lr=lr) if warmup_epochs > 0 else None
        
        print("\n" + "="*70)
        print("Training Krause-ViT" + (" (with warmup)" if warmup_epochs > 0 else ""))
        print("="*70)
        
        results = {
            'train_losses': [],
            'train_accs': [],
            'test_losses': [],
            'test_accs': []
        }
        
        best_acc = 0
        best_epoch = 0
        for epoch in range(n_epochs):
            current_warmup_scheduler = warmup_scheduler if (warmup_scheduler and warmup_scheduler.is_warming_up()) else None
            train_loss, train_acc = train_epoch(
                model, train_loader, optimizer, criterion, device,
                warmup_scheduler=current_warmup_scheduler,
                mixup_alpha=mixup_alpha,
                cutmix_alpha=cutmix_alpha,
                num_classes=model_config['num_classes'],
                mixup_prob=mixup_prob,
                label_smoothing=label_smoothing
            )
            test_loss, test_acc = evaluate(
                model, test_loader, criterion, device,
                num_classes=model_config['num_classes'],
                label_smoothing=label_smoothing
            )
            
            results['train_losses'].append(train_loss)
            results['train_accs'].append(train_acc)
            results['test_losses'].append(test_loss)
            results['test_accs'].append(test_acc)
            
            if warmup_scheduler is None or not warmup_scheduler.is_warming_up():
                scheduler.step()
            
            if test_acc > best_acc:
                best_acc = test_acc
                best_epoch = epoch + 1
                torch.save(model.state_dict(), 'best_krause_vit_cifar10_vits16.pth')
                print(f"  → Saved best Krause-ViT weights (Epoch {epoch+1}, Test Acc: {test_acc:.2f}%)")

            current_lr = optimizer.param_groups[0]['lr']
            
            sigma_values = []
            for i, block in enumerate(model.blocks):
                current_sigma = torch.exp(block.attention.log_sigma).item()
                sigma_values.append(f"L{i+1}:{current_sigma:.4f}")
            
            sigma_info = " - Sigma: " + " | ".join(sigma_values)

            if (epoch + 1) % 1 == 0:
                lr_info = f" - LR: {current_lr:.6f}" if warmup_epochs > 0 and epoch < warmup_epochs else ""
                print(f"Epoch {epoch+1}/{n_epochs} - Train: {train_acc:.2f}% - Test: {test_acc:.2f}% - Best: {best_acc:.2f}%{lr_info}{sigma_info}")
        
        print("\n" + "="*70)
        print("LOADING BEST MODEL FOR FINAL EVALUATION")
        print("="*70)

        print("\n[Loading best Krause-ViT weights]")
        model.load_state_dict(torch.load('best_krause_vit_cifar10_vits16.pth', map_location=device))
        model.eval()
        best_loss, best_acc_final = evaluate(
            model, test_loader, criterion, device,
            num_classes=model_config['num_classes'],
            label_smoothing=label_smoothing
        )
        print(f"Krause-ViT Best Model (Epoch {best_epoch}):")
        print(f"  Test Loss: {best_loss:.4f}, Test Acc: {best_acc_final:.2f}%")
        
        print("\n" + "="*70)
        print("EVALUATION SUMMARY")
        print("="*70)
        
        print("\n Classification Performance:")
        print(f"  Krause-ViT: Best Test Accuracy = {best_acc_final:.2f}%")
        
        print("\n" + "="*70)
        print("Generating Visualizations...")
        print("="*70)
        
        visualize_comprehensive_analysis(results, save_path)
        
        print("\n Evaluation complete!")
        print(f"\nGenerated file: {save_path}")
        
        return {
            'results': results,
            'best_acc': best_acc_final
        }

    print("\n" + "="*70)
    print("CONFIGURATION")
    print("="*70)
    print(f"Data path: {args.data_path}")
    print(f"Batch size: {args.batch_size}")
    print(f"Model: img_size={args.img_size}, patch_size={args.patch_size}, d_model={args.d_model}, ")
    print(f"Krause (RBF/Top-K): sigma={args.sigma}, top_k_min={args.top_k}, top_k_max={2 * args.top_k} (linear), local_coupling=4-neighbor")
    print(f"Training: epochs={args.epochs}, lr={args.lr}, weight_decay={args.weight_decay}, warmup_epochs={args.warmup_epochs}")
    print(f"drop_path={args.drop_path}, mixup_alpha={args.mixup_alpha}, cutmix_alpha={args.cutmix_alpha}, label_smoothing={args.label_smoothing}")
    print(f"Device: {'cuda' if not args.no_cuda and torch.cuda.is_available() else 'cpu'}")
    print(f"Visualization save path: {args.save_path}")
    print("="*70 + "\n")

    results = run_vit_evaluation(
        data_path=args.data_path,
        batch_size=args.batch_size,
        img_size=args.img_size,
        patch_size=args.patch_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=args.d_ff,
        dropout=args.dropout,
        sigma=args.sigma,
        top_k=args.top_k,
        n_epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        use_cuda=not args.no_cuda,
        save_path=args.save_path,
        warmup_epochs=args.warmup_epochs,
        drop_path=args.drop_path,
        mixup_alpha=args.mixup_alpha,
        cutmix_alpha=args.cutmix_alpha,
        mixup_prob=args.mixup_prob,
        label_smoothing=args.label_smoothing
    )
