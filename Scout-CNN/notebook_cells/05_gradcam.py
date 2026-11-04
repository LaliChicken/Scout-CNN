# TEACHING: weight spatial features by how much they affect this class score.
def grad_cam(model, layer, x, target=None):
    activations, gradients = [], []
    def capture(module, inputs, output):
        activations.append(output)
        output.register_hook(lambda gradient: gradients.append(gradient))
    handle = layer.register_forward_hook(capture)
    was_training = model.training
    try:
        model.eval()
        with torch.enable_grad():
            logits = model(x.detach().requires_grad_(True))  # Also works with a frozen backbone.
            target = logits.argmax(1) if target is None else target
            logits.gather(1, target[:, None]).sum().backward()
            heat = (gradients[0].mean((2, 3), keepdim=True) * activations[0]).sum(1).relu()
            heat = F.interpolate(heat[:, None], x.shape[-2:], mode="bilinear", align_corners=False)[:, 0]
            heat = heat / heat.amax((1, 2), keepdim=True).clamp_min(1e-8)
        return heat.detach(), logits.detach().softmax(1)
    finally:
        handle.remove(); model.zero_grad(set_to_none=True); model.train(was_training)
