"""Training and evaluation loops (top-1 / top-5)."""
import time

import torch
import torch.nn as nn


def _topk_correct(logits, y, ks=(1, 5)):
    maxk = min(max(ks), logits.size(1))
    top = logits.topk(maxk, dim=1).indices
    hit = top.eq(y.view(-1, 1))
    return {k: hit[:, : min(k, maxk)].any(dim=1).sum().item() for k in ks}


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    crit = nn.CrossEntropyLoss(reduction="sum")
    n, loss_sum, c1, c5 = 0, 0.0, 0, 0
    preds, targets = [], []
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        logits = model(x)
        loss_sum += crit(logits, y).item()
        c = _topk_correct(logits, y)
        c1 += c[1]
        c5 += c[5]
        n += y.size(0)
        preds += logits.argmax(1).cpu().tolist()
        targets += y.cpu().tolist()
    return {"loss": loss_sum / n, "top1": c1 / n, "top5": c5 / n,
            "preds": preds, "targets": targets}


def fit(model, loaders, device, epochs, lr=None, weight_decay=5e-4, ckpt_path="model.pt",
        momentum=0.9, label="", param_groups=None):
    """Train with SGD + cosine schedule; keep the checkpoint with best val top-1.

    Either pass `lr` (all trainable params share it) or `param_groups`
    (list of {"params": [...], "lr": ...}) for per-group learning rates.
    Parameters with requires_grad=False are never updated.
    """
    model.to(device)
    if param_groups is None:
        param_groups = [{"params": [p for p in model.parameters() if p.requires_grad], "lr": lr}]
    opt = torch.optim.SGD(param_groups, lr=param_groups[0]["lr"], momentum=momentum,
                          weight_decay=weight_decay, nesterov=True)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit = nn.CrossEntropyLoss()
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    history, best = [], -1.0
    for ep in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        loss_sum, correct, n = 0.0, 0, 0
        for x, y in loaders["train"]:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                logits = model(x)
                loss = crit(logits, y)
            if use_amp:
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                opt.step()
            loss_sum += loss.item() * y.size(0)
            correct += logits.argmax(1).eq(y).sum().item()
            n += y.size(0)
        sched.step()
        val = evaluate(model, loaders["val"], device)
        rec = {"epoch": ep, "train_loss": loss_sum / n, "train_acc": correct / n,
               "val_loss": val["loss"], "val_top1": val["top1"], "val_top5": val["top5"],
               "sec": time.time() - t0}
        history.append(rec)
        if val["top1"] > best:
            best = val["top1"]
            torch.save(model.state_dict(), ckpt_path)
        print(f"{label}ep {ep:02d}/{epochs} | "
              f"train loss {rec['train_loss']:.3f} acc {rec['train_acc']:.3f} | "
              f"val loss {rec['val_loss']:.3f} top1 {rec['val_top1']:.3f} "
              f"top5 {rec['val_top5']:.3f} | {rec['sec']:.0f}s")
    return history, best
