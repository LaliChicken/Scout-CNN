# TEACHING: predict → measure mistakes → learn. The UI calls this function.
def training_step(model, x, y, optimizer, scaler):
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(DEVICE.type, dtype=AMP_DTYPE, enabled=AMP_ENABLED):
        logits = model(x)                       # Forward: two class scores.
        loss = F.cross_entropy(logits, y)       # How wrong were we?
    scaler.scale(loss).backward()               # Gradients point toward improvement.
    scaler.step(optimizer)                      # Change the learned weights.
    scaler.update()                             # Protect small fp16 gradients.
    return loss.detach(), (logits.argmax(1) == y).sum().detach()
