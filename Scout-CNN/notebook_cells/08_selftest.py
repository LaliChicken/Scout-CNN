#@title 8 · Self-test { display-mode: "form" }
if SMOKE_TEST:
    STOP.clear()
    model = make_model("From scratch")
    cfg = dict(kind="From scratch", count=72, augment=True, size=32, epochs=1, batch_size=24, lr=0.003)
    before = model.blocks[0][0].weight.detach().clone()
    events = list(train(model, cfg))
    final = events[-1]
    assert final["steps"] == 3 and final["images_seen"] == 72
    assert not torch.equal(before, model.blocks[0][0].weight)
    assert all(key in final for key in ("epoch", "images_seen", "train_loss", "train_acc", "val_acc", "preview", "elapsed_s"))
    assert len(final["preview"]) == 8 and np.asarray(final["confusion"]).shape == (2, 2)
    assert sum(map(sum, final["confusion"])) == len(VAL_IDS)
    x = batch_images(DATA[VAL_IDS[:2]], 32)
    heat, probabilities = grad_cam(model, last_conv(model), x)
    assert heat.shape == (2, 32, 32) and probabilities.shape == (2, 2)
    assert torch.isfinite(heat).all() and 0 <= heat.min() <= heat.max() <= 1
    assert torch.allclose(probabilities.sum(1), torch.ones(2), atol=1e-5)
    assert not last_conv(model)._forward_hooks
    save_model(model, cfg, final)
    assert "loaded" in load_last()[0]
    assert test_result(None).find("Select a photo") >= 0
    assert DATA.dtype == torch.uint8 and DATA.shape == (80, 3, 144, 144)
    assert set(train_ids).isdisjoint(set(val_ids))
    print("SMOKE TEST PASSED · 3 optimizer steps · Grad-CAM · checkpoint round-trip")
