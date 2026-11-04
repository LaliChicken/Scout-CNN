"""Behavioral checks beyond the required notebook smoke test."""
from pathlib import Path
import os
import pytest
import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]

@pytest.fixture(scope="module")
def notebook():
    os.environ["SMOKE_TEST"] = "1"
    namespace = {"__name__": "__main__"}
    for path in sorted((ROOT / "notebook_cells").glob("*.py")):
        if path.name[:2] <= "06":
            exec(compile(path.read_text(), str(path), "exec"), namespace)
    return namespace

def test_exclusion_and_corruption(notebook, tmp_path):
    worker = notebook["scout_prep"]
    rng = np.random.default_rng(93)
    image = Image.fromarray(rng.integers(0, 256, (200, 170, 3), dtype=np.uint8))
    exact = tmp_path / "exact.jpg"; image.save(exact, quality=90)
    with Image.open(exact) as opened:
        ah = worker.average_hash(opened.convert("RGB"))
    sha = notebook["hashlib"].sha1(exact.read_bytes()).hexdigest()
    worker.init_worker({sha}, [ah])
    assert worker.prep_file((str(exact), 0))[0] == "exact"
    worker.init_worker(set(), [ah ^ 0b11111])
    assert worker.prep_file((str(exact), 0))[0] == "near"  # inclusive distance 5
    worker.init_worker(set(), [ah ^ 0b111111])
    result = worker.prep_file((str(exact), 0))
    assert result[0] == "ok" and result[2].shape == (144, 144, 3)
    for name, data in (("empty.jpg", b""), ("broken.jpg", b"not a photo"), ("notes.txt", b"notes")):
        path = tmp_path / name; path.write_bytes(data)
        assert worker.prep_file((str(path), 0))[0] == "invalid"

def test_frozen_transfer_and_gradcam(notebook):
    model = notebook["make_model"]("ResNet18 pretrained", pretrained=False)
    model.train()
    assert not any(m.training for m in model.modules() if isinstance(m, torch.nn.BatchNorm2d))
    assert all(not p.requires_grad for name, p in model.named_parameters() if not name.startswith("net.fc."))
    before = {name: value.clone() for name, value in model.state_dict().items()}
    x = notebook["batch_images"](notebook["DATA"][:4], 32)
    optimizer = torch.optim.AdamW(model.net.fc.parameters(), lr=.01)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    notebook["training_step"](model, x, notebook["LABELS"][:4], optimizer, scaler)
    assert not torch.equal(before["net.fc.weight"], model.net.fc.weight)
    assert all(torch.equal(before[name], value) for name, value in model.state_dict().items() if not name.startswith("net.fc."))
    layer = notebook["last_conv"](model)
    heat, probabilities = notebook["grad_cam"](model, layer, x[:1])
    assert heat.shape == (1, 32, 32) and probabilities.shape == (1, 2)
    assert torch.isfinite(heat).all() and not layer._forward_hooks
    # Hook removal also survives an invalid target.
    with pytest.raises(RuntimeError):
        notebook["grad_cam"](model, layer, x[:1], torch.tensor([9]))
    assert not layer._forward_hooks

def test_crop_flip_normalize(notebook):
    raw = notebook["DATA"][:4]
    center = notebook["batch_images"](raw, 128, False)
    expected = (raw[:, :, 8:136, 8:136].float()/255 - notebook["MEAN"]) / notebook["STD"]
    torch.testing.assert_close(center, expected)
    augmented = notebook["batch_images"](raw, 128, True)
    assert augmented.shape == center.shape and torch.isfinite(augmented).all()
    assert not torch.equal(center, augmented)

def test_stop_and_no_model(notebook):
    notebook["MODEL"] = None
    assert "Train or load" in notebook["test_result"](None)
    assert "Train or load" in notebook["inside"](None)[0]
    notebook["STOP"].set()
    cfg = dict(kind="From scratch", count=72, augment=False, size=32, epochs=3, batch_size=24, lr=.003)
    events = list(notebook["train"](notebook["make_model"]("From scratch"), cfg))
    assert len(events) == 1 and events[0]["stopped"] and events[0]["steps"] == 0
    assert events[0]["final"] and sum(map(sum, events[0]["confusion"])) == 8
    notebook["STOP"].clear()

def test_safe_archive(notebook, tmp_path):
    import zipfile
    path = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("../outside.txt", "no")
    with pytest.raises(ValueError):
        notebook["safe_extract"](path, tmp_path / "extract")
    assert not (tmp_path / "outside.txt").exists()
