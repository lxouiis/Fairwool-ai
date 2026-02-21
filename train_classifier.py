import os
from dataclasses import dataclass
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
import timm

@dataclass
class TrainConfig:
    model_name: str = "deit_small_patch16_224.fb_in1k"
    num_classes: int = 3
    image_size: int = 224
    batch_size: int = 16 # reduce to 8 if you hit OOM on MPS
    num_epochs_head: int = 5
    num_epochs_finetune: int = 8
    lr_head: float = 2e-3
    lr_finetune: float = 5e-5
    weight_decay: float = 0.02
    num_workers: int = 2
    seed: int = 42

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def set_seed(seed: int):
    torch.manual_seed(seed)
    # MPS uses torch.mps RNG behind the scenes; torch.manual_seed is still the standard entry point.

def build_loaders(data_root: str, image_size: int, batch_size: int, num_workers: int):
    # For hackathon simplicity, use ImageNet mean/std (common choice).
    # You can also swap this with timm.data.create_transform() (see below).
    train_tfms = transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.65, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10, hue=0.02),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

    eval_tfms = transforms.Compose([
        transforms.Resize(int(image_size * 1.15)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

    # Check if directories exist before creating ImageFolder to avoid crashes on empty/missing dirs
    train_dir = os.path.join(data_root, "train")
    val_dir = os.path.join(data_root, "val")
    
    if not os.path.exists(train_dir):
        raise FileNotFoundError(f"Training directory not found: {train_dir}")
    if not os.path.exists(val_dir):
        # Fallback if val doesn't exist, maybe split train? Or just warn.
        # For this script, we assume strict structure.
        raise FileNotFoundError(f"Validation directory not found: {val_dir}")

    train_ds = ImageFolder(train_dir, transform=train_tfms)
    val_ds = ImageFolder(val_dir, transform=eval_tfms)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, val_loader, train_ds.class_to_idx

def build_model(model_name: str, num_classes: int, device: torch.device):
    model = timm.create_model(model_name, pretrained=True)
    # timm provides a consistent reset method for many models
    if hasattr(model, "reset_classifier"):
        model.reset_classifier(num_classes=num_classes)
    else:
        # fallback: replace common head attribute
        in_features = model.get_classifier().in_features
        model.set_classifier(nn.Linear(in_features, num_classes))
    model.to(device)
    return model

def freeze_all_but_head(model: nn.Module):
    for p in model.parameters():
        p.requires_grad = False
    
    # timm convention: classifier is often 'head'
    head_params = []
    if hasattr(model, "head"):
        for p in model.head.parameters():
            p.requires_grad = True
            head_params.append(p)
    else:
        # fallback: train any params that are still requires_grad
        head_params = [p for p in model.parameters() if p.requires_grad]
    
    return head_params

def unfreeze_last_transformer_blocks(model: nn.Module, n_blocks: int = 1):
    """
    Works for DeiT/ViT-style timm models that expose model.blocks as a list/ModuleList.
    """
    if not hasattr(model, "blocks"):
        return []
    
    for blk in model.blocks[-n_blocks:]:
        for p in blk.parameters():
            p.requires_grad = True
    
    # return trainable params to pass to optimizer
    return [p for p in model.parameters() if p.requires_grad]

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return correct / max(total, 1)

def train_one_epoch(model, loader, optimizer, loss_fn, device, use_amp: bool):
    model.train()
    running_loss = 0.0
    
    # scaler = torch.cuda.amp.GradScaler() if use_amp else None # Not used for MPS currently in snippet
    
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        
        # AMP note: torch.autocast supports device_type strings.
        if use_amp:
             with torch.autocast(device_type=device.type, dtype=torch.float16):
                logits = model(x)
                loss = loss_fn(logits, y)
        else:
            logits = model(x)
            loss = loss_fn(logits, y)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * x.size(0)
    
    return running_loss / len(loader.dataset)

def main(data_root="data", out_path="fairwool_deit_small.pt"):
    cfg = TrainConfig()
    set_seed(cfg.seed)
    device = get_device()
    
    # Optional: if you want to cap MPS memory usage (useful on smaller unified-memory Macs)
    # torch.mps.set_per_process_memory_fraction(0.8) # Example
    
    print(f"Using device: {device}")

    if not os.path.exists(data_root):
        print(f"Data root '{data_root}' does not exist. Please create 'data/train' and 'data/val' first.")
        # Create dummy structure for user to fill?
        return

    train_loader, val_loader, class_to_idx = build_loaders(
        data_root, cfg.image_size, cfg.batch_size, cfg.num_workers
    )
    
    model = build_model(cfg.model_name, cfg.num_classes, device)
    
    # Whether AMP/autocast is available on this device type can be checked.
    use_amp = False
    try:
        if hasattr(torch.amp, 'autocast_mode'): # PyTorch 2.x
             use_amp = torch.amp.autocast_mode.is_autocast_available(device.type)
        else:
             use_amp = False # Safer default if unsure about version/MPS support
    except Exception:
        use_amp = False
    
    print(f"AMP enabled: {use_amp}")

    # Stage 1: train head only
    print("Stage 1: Training Head...")
    head_params = freeze_all_but_head(model)
    optimizer = torch.optim.AdamW(head_params, lr=cfg.lr_head, weight_decay=cfg.weight_decay)
    loss_fn = nn.CrossEntropyLoss()
    
    best_val = 0.0
    for epoch in range(cfg.num_epochs_head):
        tr_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device, use_amp=use_amp)
        val_acc = evaluate(model, val_loader, device)
        best_val = max(best_val, val_acc)
        print(f"[Head] epoch={epoch+1} loss={tr_loss:.4f} val_acc={val_acc:.3f}")
    
    # Stage 2: unfreeze last transformer block(s) and fine-tune with smaller LR
    print("Stage 2: Fine-tuning...")
    trainable_params = unfreeze_last_transformer_blocks(model, n_blocks=1)
    optimizer = torch.optim.AdamW(trainable_params, lr=cfg.lr_finetune, weight_decay=cfg.weight_decay)
    
    for epoch in range(cfg.num_epochs_finetune):
        tr_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device, use_amp=use_amp)
        val_acc = evaluate(model, val_loader, device)
        best_val = max(best_val, val_acc)
        print(f"[FT] epoch={epoch+1} loss={tr_loss:.4f} val_acc={val_acc:.3f}")
        
    ckpt = {
        "model_name": cfg.model_name,
        "class_to_idx": class_to_idx,
        "state_dict": model.state_dict(),
        "image_size": cfg.image_size,
    }
    torch.save(ckpt, out_path)
    print("Saved:", out_path, "best_val_acc:", best_val)

if __name__ == "__main__":
    main()
