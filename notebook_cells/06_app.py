#@title 6 · Training and app { display-mode: "form" }
from torchvision.models import resnet18, ResNet18_Weights
from sklearn.decomposition import PCA
from contextlib import contextmanager

MODEL, MODEL_CFG, MODEL_VERSION = None, None, 0
STOP = threading.Event()
INSIDE_CACHE = {}
MEAN = torch.tensor([0.485, 0.456, 0.406], device=DEVICE)[None, :, None, None]
STD = torch.tensor([0.229, 0.224, 0.225], device=DEVICE)[None, :, None, None]

class TransferCNN(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.net = resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
        for parameter in self.net.parameters():
            parameter.requires_grad_(False)
        self.net.fc = nn.Linear(self.net.fc.in_features, 2)
    def train(self, mode=True):
        super().train(mode)
        self.net.eval()  # Frozen BatchNorm statistics as well as frozen weights.
        self.net.fc.train(mode)
        return self
    def forward(self, x):
        return self.net(x)

def make_model(kind, pretrained=True):
    model = SmallCNN() if kind == "From scratch" else TransferCNN(pretrained)
    return model.to(DEVICE, memory_format=torch.channels_last)

def last_conv(model):
    return model.blocks[-1][3] if isinstance(model, SmallCNN) else model.net.layer4[-1].conv2

def last_block(model):
    return model.blocks[-1] if isinstance(model, SmallCNN) else model.net.layer4

def batch_images(raw, size, augment=False):
    n, _, h, w = raw.shape
    if augment:
        top = torch.randint(h - size + 1, (n, 1, 1), device=raw.device)
        left = torch.randint(w - size + 1, (n, 1, 1), device=raw.device)
        rows = top + torch.arange(size, device=raw.device)[None, :, None]
        cols = left + torch.arange(size, device=raw.device)[None, None, :]
        x = raw.permute(0, 2, 3, 1)[torch.arange(n, device=raw.device)[:, None, None], rows, cols].permute(0, 3, 1, 2)
        flip = torch.rand(n, 1, 1, 1, device=raw.device) < 0.5
        x = torch.where(flip, x.flip(-1), x)
    else:
        top, left = (h - size) // 2, (w - size) // 2
        x = raw[:, :, top:top+size, left:left+size]
    return ((x.float() / 255 - MEAN) / STD).contiguous(memory_format=torch.channels_last)

@contextmanager
def evaluating(model):
    previous = model.training
    model.eval()
    try:
        with torch.no_grad():
            yield
    finally:
        model.train(previous)

def predict_ids(model, ids, size, batch_size=256):
    probabilities = []
    with evaluating(model):
        for indices in ids.split(batch_size):
            with torch.autocast(DEVICE.type, dtype=AMP_DTYPE, enabled=AMP_ENABLED):
                probabilities.append(model(batch_images(DATA[indices], size)).float().softmax(1))
    return torch.cat(probabilities)

def validate(model, size):
    probabilities = predict_ids(model, VAL_IDS, size)
    predicted, actual = probabilities.argmax(1), LABELS[VAL_IDS]
    matrix = torch.bincount(actual * 2 + predicted, minlength=4).reshape(2, 2).cpu().tolist()
    return (predicted == actual).float().mean().item(), matrix

def preview_predictions(model, size):
    probs = predict_ids(model, PREVIEW_IDS, size)
    confidence, prediction = probs.max(1)
    return [{"label": CLASS_NAMES[p], "confidence": c, "correct": p == truth}
            for p, c, truth in zip(prediction.cpu().tolist(), confidence.cpu().tolist(), LABELS[PREVIEW_IDS].cpu().tolist())]

def balanced_subset(count):
    generator = torch.Generator(device=DEVICE).manual_seed(SEED)
    selected = []
    for label in (0, 1):
        available = TRAIN_IDS[LABELS[TRAIN_IDS] == label]
        order = torch.randperm(len(available), generator=generator, device=DEVICE)
        selected.append(available[order[:min(len(available), count // 2 + (label == 0 and count % 2))]])
    return torch.cat(selected)

def train(model, cfg):
    """The UI and smoke test consume this generator; all computation uses GPU tensors."""
    indices = balanced_subset(cfg["count"])
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=cfg["lr"], weight_decay=1e-3)
    steps_per_epoch = math.ceil(len(indices) / cfg["batch_size"])
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=cfg["lr"], total_steps=steps_per_epoch * cfg["epochs"], pct_start=0.3)
    scaler = torch.amp.GradScaler("cuda", enabled=AMP_ENABLED and AMP_DTYPE == torch.float16)
    started = time.perf_counter()
    seen, step, val_acc, confusion = 0, 0, None, None
    total_loss = torch.zeros((), device=DEVICE)
    total_correct = torch.zeros((), device=DEVICE)
    def event(epoch, epoch_seen, final=False, epoch_end=False):
        nonlocal val_acc, confusion
        preview = preview_predictions(model, cfg["size"])
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        return {"epoch": epoch, "images_seen": seen, "train_loss": total_loss.item() / max(1, epoch_seen),
                "train_acc": total_correct.item() / max(1, epoch_seen), "val_acc": val_acc,
                "preview": preview, "elapsed_s": time.perf_counter() - started,
                "epoch_end": epoch_end, "confusion": confusion if final else None, "final": final, "stopped": STOP.is_set(), "steps": step}
    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        total_loss.zero_(); total_correct.zero_(); epoch_seen = 0
        order = indices[torch.randperm(len(indices), device=DEVICE)]
        for ids in order.split(cfg["batch_size"]):
            if STOP.is_set():
                break
            x = batch_images(DATA[ids], cfg["size"], cfg["augment"])
            loss, correct = training_step(model, x, LABELS[ids], optimizer, scaler)
            scheduler.step()
            total_loss += loss * len(ids); total_correct += correct
            epoch_seen += len(ids); seen += len(ids); step += 1
            if step % 25 == 0:
                yield event(epoch, epoch_seen)
        val_acc, confusion = validate(model, cfg["size"])
        final = STOP.is_set() or epoch == cfg["epochs"]
        yield event(epoch, epoch_seen, final, epoch_end=True)
        if final:
            break
    model.eval()

def save_model(model, cfg, final):
    checkpoint = {"state_dict": model.state_dict(), "cfg": cfg, "metrics": final, "version": 1}
    temporary = CHECKPOINTS / "last.tmp"
    torch.save(checkpoint, temporary)
    temporary.replace(CHECKPOINTS / "last.pt")
    report = {"gpu": torch.cuda.get_device_name(0) if DEVICE.type == "cuda" else "CPU synthetic smoke",
              "synthetic": SMOKE_TEST, "stopped": final["stopped"], "last_epoch": final["epoch"],
              "images_seen": final["images_seen"], "prep_s": PREP_SECONDS, "training_s": final["elapsed_s"],
              "val_accuracy": final["val_acc"], "cfg": cfg, "audit": audit}
    (CHECKPOINTS / "last_run.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k not in ("cfg", "audit")}, indent=2))

# Inline PNG/SVG outputs avoid Gradio's image-file cache and expose no source filenames.
def image_uri(array):
    buffer = io.BytesIO()
    Image.fromarray(np.asarray(array, dtype=np.uint8)).save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()

def figure_html(fig, height):
    buffer = io.StringIO()
    fig.savefig(buffer, format="svg", facecolor="white")
    plt.close(fig)
    svg = buffer.getvalue()[buffer.getvalue().index("<svg"):]
    return f'<div class="chart" style="height:{height}px">{svg}</div>'

plt.rcParams.update({"font.size": 17, "axes.spines.top": False, "axes.spines.right": False,
                     "svg.fonttype": "none", "axes.labelcolor": "#14243b", "text.color": "#14243b"})

def charts(history):
    fig, ax = plt.subplots(figsize=(11.5, 3.5), layout="constrained")
    ax.set_ylim(0, 105); ax.set_ylabel("Accuracy %"); ax.set_xlabel("Images seen")
    ax.axhline(50, color="#65758b", linestyle="--", linewidth=2, label="Chance")
    if history:
        x = [e["images_seen"] for e in history]
        ax.plot(x, [e["train_acc"] * 100 for e in history], color="#075eae", lw=3, marker="o", label="Train")
        valid = [e for e in history if e["val_acc"] is not None and e.get("epoch_end", False)]
        ax.plot([e["images_seen"] for e in valid], [e["val_acc"] * 100 for e in valid], color="#c34c00", marker="o", lw=3, label="Validation")
    ax.legend(loc="lower right", ncol=3, fontsize=16); ax.grid(alpha=0.15)
    accuracy = figure_html(fig, 310)
    fig, ax = plt.subplots(figsize=(11.5, 1.75), layout="constrained")
    if history:
        ax.plot(x, [e["train_loss"] for e in history], color="#573aa3", lw=3, marker="o")
    ax.set_ylabel("Loss"); ax.set_xlabel("Images seen"); ax.grid(alpha=0.15)
    return accuracy, figure_html(fig, 160)

def previews_html(preview=None):
    cards = []
    for i, idx in enumerate(PREVIEW_IDS.cpu().tolist()):
        if preview:
            result = preview[i]
            color = "#15803d" if result["correct"] else "#c62828"
            text = f'<div class="preview-label">{result["label"]}</div><div class="preview-number">{result["confidence"]:.0%}</div>'
        else:
            color, text = "#b8c4d4", "Ready"
        cards.append(f'<div class="preview"><img alt="Validation photo" src="{image_uri(pixels[idx])}" style="border-color:{color}"><div>{text}</div></div>')
    return '<div class="previews">' + ''.join(cards) + '</div>'

def matrix_html(matrix=None):
    if matrix is None:
        return '<div class="matrix empty"><b>Confusion matrix</b><div>After training</div></div>'
    return f'''<div class="matrix"><b>Confusion matrix</b><div>Predicted</div><table>
    <tr><th>Truth</th><th>CAT</th><th>DOG</th></tr>
    <tr><th>CAT</th><td class="correct">{matrix[0][0]}</td><td>{matrix[0][1]}</td></tr>
    <tr><th>DOG</th><td>{matrix[1][0]}</td><td class="correct">{matrix[1][1]}</td></tr></table></div>'''

def train_ui(kind, count, augment):
    global MODEL, MODEL_CFG, MODEL_VERSION
    STOP.clear(); INSIDE_CACHE.clear()
    MODEL = make_model(kind)
    MODEL_VERSION += 1
    batch_size = 256 if DEVICE.type == "cuda" and torch.cuda.get_device_properties(0).total_memory >= 30 * 2**30 else 128
    cfg = dict(kind=kind, count=int(count), augment=bool(augment), size=img_size, epochs=max(epochs, math.ceil(40000 / int(count))) if int(count) <= 2000 else epochs,
               batch_size=batch_size, lr=0.003 if kind == "From scratch" else 0.01)
    if SMOKE_TEST:
        cfg.update(epochs=1, batch_size=24, size=img_size, count=72)
    MODEL_CFG = cfg
    history = []
    parameters = sum(p.numel() for p in MODEL.parameters()) / 1e6
    yield *charts([]), previews_html(), '<div class="status">Starting</div>', matrix_html()
    try:
        for event in train(MODEL, cfg):
            history.append(event)
            val = "—" if event["val_acc"] is None else f'{event["val_acc"]:.1%}'
            status = f'Epoch {event["epoch"]}/{cfg["epochs"]} · {event["images_seen"]:,} images · val {val} · {parameters:.2f}M params'
            if event["final"]:
                status += f' · {event["elapsed_s"]:.0f}s' + (" · Stopped" if event["stopped"] else "")
                save_model(MODEL, cfg, event)
            yield *charts(history), previews_html(event["preview"]), f'<div class="status">{status}</div>', matrix_html(event["confusion"])
    except torch.cuda.OutOfMemoryError:
        raise gr.Error("GPU memory full. Restart the runtime or lower Training images in Config.")

def stop_training():
    STOP.set()

def load_last():
    global MODEL, MODEL_CFG, MODEL_VERSION
    path = CHECKPOINTS / "last.pt"
    if not path.exists():
        return '<div class="status">No saved model</div>', matrix_html()
    checkpoint = torch.load(path, map_location=DEVICE, weights_only=True)
    MODEL_CFG = checkpoint["cfg"]
    MODEL = make_model(MODEL_CFG["kind"], pretrained=False)
    MODEL.load_state_dict(checkpoint["state_dict"]); MODEL.eval()
    MODEL_VERSION += 1; INSIDE_CACHE.clear()
    return '<div class="status">Last model loaded</div>', matrix_html(checkpoint["metrics"]["confusion"])

NO_MODEL = '<div class="empty">Train or load a model first</div>'

def demo_gallery(difficulty="All", reveal=False, page=0):
    matches = [d for d in DEMOS if difficulty == "All" or d["difficulty"] == difficulty.lower()]
    pages = max(1, math.ceil(len(matches) / 6)); page = int(page) % pages
    cards = []
    for d in matches[page*6:(page+1)*6]:
        label = CLASS_NAMES[d["label"]] if reveal else "Select"
        cards.append(f'<button class="demo-pick" data-demo="{d["id"]}" aria-label="Select photo {d["id"]+1}"><img alt="Demo photo" src="{image_uri(d["image"])}"><span>{label}</span></button>')
    return '<div class="demo-grid">' + ''.join(cards) + f'</div><div class="page-label">Page {page+1} / {pages}</div>', page

def photo_tensor(array, size):
    raw = torch.from_numpy(scout_prep.canonical(Image.fromarray(array))).permute(2, 0, 1)[None].to(DEVICE)
    return batch_images(raw, size)

def test_result(selected, reveal=False):
    if MODEL is None:
        return NO_MODEL
    if selected is None:
        return '<div class="empty">Select a photo</div>'
    array, truth = selected["image"], selected.get("truth")
    size = MODEL_CFG["size"]
    x = photo_tensor(array, size)
    heat, probabilities = grad_cam(MODEL, last_conv(MODEL), x)
    probability, predicted = probabilities[0].max(0)
    display_photo = ((x[0] * STD[0] + MEAN[0]).permute(1, 2, 0).cpu().numpy() * 255).round().clip(0, 255).astype(np.uint8)
    colors = plt.get_cmap("jet")(heat[0].cpu().numpy())[:, :, :3]
    overlay = (0.55 * display_photo + 0.45 * colors * 255).clip(0, 255).astype(np.uint8)
    verdict, confidence = CLASS_NAMES[predicted.item()], probability.item() * 100
    truth_text = CLASS_NAMES[truth] if reveal and truth is not None else ("Unknown" if reveal else "Hidden")
    return f'''<div class="test-result"><div class="photo-pair">
    <div><h3>Photo</h3><img alt="Selected photo" src="{image_uri(display_photo)}"></div>
    <div><h3>Grad-CAM</h3><img alt="Grad-CAM overlay" src="{image_uri(overlay)}"></div></div>
    <div class="prediction"><div class="verdict">{verdict}</div><div class="confidence">{confidence:.1f}%</div>
    <div class="confidence-track" role="meter" aria-label="Confidence" aria-valuenow="{confidence:.1f}" aria-valuemin="0" aria-valuemax="100"><div style="width:{confidence:.1f}%"></div></div><div>Truth: {truth_text}</div></div></div>'''

def select_demo(index, reveal):
    try:
        index = int(index)
        assert 0 <= index < len(DEMOS)
    except (ValueError, AssertionError):
        raise gr.Error("Select a photo")
    d = DEMOS[index]
    selected = {"image": d["image"], "truth": d["label"]}
    return selected, test_result(selected, reveal)

def select_upload(encoded, reveal):
    if not encoded:
        return None, '<div class="empty">Choose a photo</div>', ""
    if len(encoded) > 3_000_000 or not encoded.startswith("data:image/jpeg;base64,"):
        raise gr.Error("Invalid image")
    # Client resizes to <=512 px. Decoding stays in RAM; no file uploader or logging.
    with Image.open(io.BytesIO(base64.b64decode(encoded.split(",", 1)[1], validate=True))) as opened:
        if max(opened.size) > 512:
            raise gr.Error("Image too large")
        array = np.asarray(opened.convert("RGB")).copy()
    selected = {"image": array, "truth": None}
    return selected, test_result(selected, reveal), ""

def filters_html():
    first = MODEL.blocks[0][0] if isinstance(MODEL, SmallCNN) else MODEL.net.conv1
    weights = first.weight.detach().float().cpu()
    # Per-filter rescaling shows the pattern, not absolute magnitude.
    lo = weights.amin((1, 2, 3), keepdim=True); hi = weights.amax((1, 2, 3), keepdim=True)
    tiles = ((weights-lo) / (hi-lo).clamp_min(1e-8) * 255).byte().permute(0, 2, 3, 1).numpy()
    return '<h3>Learned filters</h3><div class="filter-grid">' + ''.join(f'<img alt="Learned filter" src="{image_uri(tile)}">' for tile in tiles) + '</div>'

def maps_html(selected):
    if selected is None:
        return '<h3>Feature maps</h3><div class="empty">Select a photo</div>'
    features = []
    hook = last_block(MODEL).register_forward_hook(lambda module, inputs, output: features.append(output.detach()))
    try:
        with evaluating(MODEL):
            MODEL(photo_tensor(selected["image"], MODEL_CFG["size"]))
    finally:
        hook.remove()
    # Show the 16 most active channels, from the last block.
    values = features[0][0].float()
    top = values.mean((1, 2)).topk(16).indices
    values = values[top]
    values = values / values.amax((1, 2), keepdim=True).clamp_min(1e-8)
    tiles = (plt.get_cmap("magma")(values.cpu().numpy())[:, :, :, :3] * 255).astype(np.uint8)
    return '<h3>Feature maps</h3><div class="maps-grid">' + ''.join(f'<img alt="Feature map" src="{image_uri(tile)}">' for tile in tiles) + '</div>'

def embeddings(raw, size):
    captured = []
    layer = MODEL.head[1] if isinstance(MODEL, SmallCNN) else MODEL.net.avgpool
    hook = layer.register_forward_hook(lambda module, inputs, output: captured.append(output.flatten(1).detach().float().cpu()))
    try:
        with evaluating(MODEL):
            for batch in raw.split(128):
                MODEL(batch_images(batch, size))
    finally:
        hook.remove()
    return torch.cat(captured).numpy()

def pca_html():
    if MODEL_VERSION in INSIDE_CACHE:
        return INSIDE_CACHE[MODEL_VERSION]
    ids = VAL_IDS[:min(1000, len(VAL_IDS))]
    # Fit only on validation embeddings; held-out stars never influence the PCA axes.
    z = embeddings(DATA[ids], MODEL_CFG["size"])
    pca = PCA(n_components=2, svd_solver="full").fit(z)
    points = pca.transform(z)
    tricky = [d for d in DEMOS if d["difficulty"] == "tricky"]
    raw = torch.from_numpy(np.stack([d["image"] for d in tricky])).permute(0, 3, 1, 2).to(DEVICE)
    stars = pca.transform(embeddings(raw, MODEL_CFG["size"]))
    actual = LABELS[ids].cpu().numpy()
    fig, ax = plt.subplots(figsize=(8.3, 6.5), layout="constrained")
    for label, color in enumerate(("#075eae", "#c34c00")):
        mask = actual == label
        ax.scatter(points[mask, 0], points[mask, 1], s=26, alpha=0.4, c=color, label=CLASS_NAMES[label])
    span = np.maximum(np.ptp(np.vstack([points, stars]), axis=0), 1e-3)
    for i, (point, d) in enumerate(zip(stars, tricky)):
        color = "#075eae" if d["label"] == 0 else "#c34c00"
        ax.scatter(*point, marker="*", s=240, c=color, edgecolors="#111827", linewidths=1, zorder=4)
        # Spread labels into two gutters with leader lines; all 16 remain legible.
        side = i % 2
        anchor = (points[:, 0].min() - span[0]*0.28 if side == 0 else points[:, 0].max()+span[0]*0.28,
                  points[:, 1].min() + span[1]*(i//2)/7)
        ax.annotate(f'{CLASS_NAMES[d["label"]][0]}{i+1}', xy=point, xytext=anchor,
                    fontsize=17, ha="right" if side == 0 else "left", va="center",
                    arrowprops={"arrowstyle": "-", "color": color, "alpha": 0.65}, color=color)
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(4))
    ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(5))
    ax.set_xlabel("PCA 1"); ax.set_ylabel("PCA 2"); ax.legend(loc="upper center", ncol=2)
    ax.margins(x=0.4, y=0.15); ax.set_title(f"{len(ids):,} validation · 16 tricky stars", fontsize=19)
    result = '<h3>Feature space</h3>' + figure_html(fig, 630)
    INSIDE_CACHE[MODEL_VERSION] = result
    return result

def inside(selected):
    if MODEL is None:
        return NO_MODEL, "", ""
    return filters_html(), maps_html(selected), pca_html()

CSS = """
:root { color-scheme: light !important; }
body, .gradio-container { background: #f2f5fa !important; color: #14243b !important; }
.gradio-container { width: calc(100% - 40px) !important; max-width: 1880px !important; box-sizing:border-box; margin: 0 auto !important; padding: 18px 20px !important; }
#training-count .head { flex-direction:column; align-items:stretch; gap:8px; }
#training-count .tab-like-container { width:100%; min-height:60px; }
#training-count input[type="number"] { font-size:48px !important; font-weight:800; width:100% !important; max-width:none !important; }
#training-count .reset-button { display:none !important; }
.gradio-container main { width:100% !important; padding:0 !important; }
.gradio-container [data-testid="block-info"] { font-size:22px !important; }
.gradio-container, .gradio-container * { font-family: Arial, sans-serif; font-size: 22px; }
.gradio-container button, .capture-label { min-height: 64px !important; font-size: 24px !important; font-weight: 700 !important; border-radius: 10px !important; }
.gradio-container input, .gradio-container select, .gradio-container textarea { font-size: 22px !important; }
.gradio-container label, .gradio-container label span { font-size: 22px !important; }
footer, a[href*="gradio.app"], .api-docs, .sr-data { display: none !important; }
.title { display:flex; align-items:baseline; gap:22px; margin:0 0 10px; }
.title strong { font-size:38px; } .title span { color:#425775; font-size:26px; }
.tab-nav { margin-bottom:12px !important; border-bottom:2px solid #c2cedd !important; }
.tab-nav button.selected { background:#173e71 !important; color:white !important; }
.chart svg { width:100%; height:100%; display:block; }
.chart { background:white; border-radius:12px; }
.status { padding:12px 16px; background:#e4edf9; color:#153b68; border-radius:10px; font-size:26px; font-weight:700; white-space:nowrap; }
h3 { font-size:26px !important; font-weight:700; margin:0 0 14px !important; }
.previews { display:grid; grid-template-columns:repeat(8,minmax(0,1fr)); gap:10px; }
.preview { text-align:center; background:white; border-radius:10px; padding:5px; }
.preview img { display:block; width:100%; height:126px; object-fit:cover; border:6px solid; border-radius:8px; box-sizing:border-box; }
.preview .preview-label { font-size:22px; line-height:26px; } .preview .preview-number { font-size:48px; line-height:52px; font-weight:800; }
.matrix { padding:18px 8px; text-align:center; background:white; border-radius:12px; }
.matrix b { font-size:24px; display:block; margin-bottom:16px; }
.matrix table { border-spacing:5px; width:100%; margin-top:15px; }
.matrix th { font-size:22px; padding:5px; } .matrix td { background:#ffe7e2; font-size:48px; font-weight:800; padding:12px 3px; }
.matrix td.correct { background:#daf0e0; color:#14532d; }
.empty { display:flex; flex-direction:column; justify-content:center; align-items:center; min-height:250px; font-size:30px; color:#425775; text-align:center; }
.demo-grid { display:grid; grid-template-columns:repeat(2,1fr); gap:8px; }
.demo-pick { border:2px solid #b7c8dd; background:white; padding:4px; cursor:pointer; }
.demo-pick:hover, .demo-pick:focus { outline:4px solid #075eae; }
.demo-pick img { width:100%; height:83px; object-fit:cover; display:block; border-radius:6px; }
.demo-pick span { display:block; font-size:22px; line-height:26px; }
.page-label { text-align:center; padding-top:10px; }
.test-result { display:grid; grid-template-columns:minmax(0,1fr) 235px; gap:22px; align-items:center; padding:20px 8px; }
.photo-pair { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:18px; }
.photo-pair img { width:100%; aspect-ratio:1; object-fit:contain; background:white; border-radius:16px; }
.verdict { font-size:72px; font-weight:900; color:#153b68; } .confidence { font-size:56px; font-weight:800; margin:16px 0; }
.confidence-track { height:34px; background:#d1dfef; border-radius:8px; margin-bottom:24px; overflow:hidden; }
.confidence-track div { height:100%; background:#075eae; }
.filter-grid { display:grid; grid-template-columns:repeat(8,1fr); gap:8px; }
.filter-grid img { width:100%; aspect-ratio:1; image-rendering:pixelated; border:1px solid #b7c8dd; }
.maps-grid { display:grid; grid-template-columns:repeat(4,1fr); gap:10px; }
.maps-grid img { width:100%; aspect-ratio:1; image-rendering:pixelated; }
#capture-video { display:block; width:100%; height:260px; object-fit:contain; background:#14243b; border-radius:12px; margin:12px 0; }
.capture-actions { display:flex; gap:8px; } .capture-actions button, .capture-label { flex:1; background:#e4edf9; color:#14243b; border:2px solid #a9bbd2; padding:12px; text-align:center; cursor:pointer; }
.capture-label { display:block; box-sizing:border-box; } #capture-file { display:none; }
#capture-status { min-height:32px; margin-top:8px; }
"""

# Files and webcam frames are decoded in the browser, resized, then sent as text.
# No gr.Image input / /upload endpoint is used. No raw image files are written.
CLIENT_JS = r"""() => {
  if (window.scoutEvents) return;
  window.scoutEvents = true;
  let stream = null;
  const stop = () => { if(stream) stream.getTracks().forEach(t=>t.stop()); stream=null; };
  const setText = (id, value) => {
    const input = document.querySelector('#'+id+' textarea, #'+id+' input');
    const proto = input.tagName==='TEXTAREA' ? HTMLTextAreaElement.prototype : HTMLInputElement.prototype;
    Object.getOwnPropertyDescriptor(proto,'value').set.call(input,value);
    input.dispatchEvent(new Event('input',{bubbles:true}));
  };
  const status = text => { const el=document.querySelector('#capture-status'); if(el) el.textContent=text; };
  const send = (source,w,h) => {
    const scale=Math.min(1,512/Math.max(w,h));
    const canvas=document.createElement('canvas');
    canvas.width=Math.round(w*scale); canvas.height=Math.round(h*scale);
    canvas.getContext('2d').drawImage(source,0,0,canvas.width,canvas.height);
    setText('capture-data',canvas.toDataURL('image/jpeg',0.88));
    requestAnimationFrame(()=>document.querySelector('#capture-submit').click());
    canvas.width=0; canvas.height=0;
    stop(); status('Photo ready');
  };
  document.addEventListener('click',async event => {
    const demo=event.target.closest('.demo-pick');
    if(demo) {setText('demo-index',demo.dataset.demo); requestAnimationFrame(()=>document.querySelector('#demo-submit').click());}
    if(event.target.closest('#camera-start')) {
      try {stop(); stream=await navigator.mediaDevices.getUserMedia({video:true,audio:false});
        const video=document.querySelector('#capture-video'); video.srcObject=stream; await video.play(); status('Camera ready');
      } catch(error) {status('Camera unavailable');}
    }
    if(event.target.closest('#camera-capture')) {
      const video=document.querySelector('#capture-video');
      if(video.videoWidth) send(video,video.videoWidth,video.videoHeight); else status('Start camera first');
    }
  });
  document.addEventListener('change',async event=>{
    if(event.target.id==='capture-file') {
      const file=event.target.files[0]; if(!file) return;
      try {const bitmap=await createImageBitmap(file); send(bitmap,bitmap.width,bitmap.height); bitmap.close();}
      catch(error) {status('Choose an image');}
      event.target.value='';
    }
    if(event.target.type==='radio' && event.target.value==='Demo photos') stop();
  });
  window.addEventListener('pagehide',stop);
}"""

CAPTURE_HTML = '''<label class="capture-label" for="capture-file">Upload photo</label><input id="capture-file" type="file" accept="image/*">
<video id="capture-video" autoplay playsinline muted></video><div class="capture-actions">
<button type="button" id="camera-start">Camera</button><button type="button" id="camera-capture">Capture</button></div>
<div id="capture-status" role="status"></div>'''

with gr.Blocks(title="Scout AI: Cats vs Dogs", css=CSS, js=CLIENT_JS,
               theme=gr.themes.Base(primary_hue="blue", neutral_hue="slate"), analytics_enabled=False) as app:
    gr.HTML('<div class="title"><strong>Scout AI</strong><span>Cats vs Dogs</span></div>')
    selected = gr.State(None, time_to_live=3600)
    page = gr.State(0)
    with gr.Tabs():
        with gr.Tab("Train"):
            with gr.Row():
                with gr.Column(scale=1, min_width=310):
                    model_choice = gr.Radio(["From scratch", "ResNet18 pretrained"], value="From scratch", label="Model")
                    count = gr.Slider(500 if not SMOKE_TEST else 24, len(TRAIN_IDS), value=len(TRAIN_IDS), step=1, label="Training images", elem_id="training-count")
                    augment = gr.Checkbox(True, label="Augmentation")
                    start = gr.Button("Start", variant="primary")
                    stop = gr.Button("Stop")
                    load = gr.Button("Load last model")
                with gr.Column(scale=5, min_width=1000):
                    with gr.Row():
                        with gr.Column(scale=4, min_width=650):
                            initial_accuracy, initial_loss = charts([])
                            acc_plot = gr.HTML(initial_accuracy)
                            loss_plot = gr.HTML(initial_loss)
                        with gr.Column(scale=1, min_width=360):
                            confusion_plot = gr.HTML(matrix_html())
                    preview_grid = gr.HTML(previews_html())
                    train_status = gr.HTML('<div class="status">Ready to learn</div>')
        with gr.Tab("Test") as test_tab:
            with gr.Row():
                with gr.Column(scale=1, min_width=420):
                    source = gr.Radio(["Demo photos", "Camera/Upload"], value="Demo photos", label="Source")
                    reveal = gr.Checkbox(False, label="Reveal")
                    with gr.Group() as demo_group:
                        difficulty = gr.Radio(["All", "Easy", "Tricky"], value="All", label="Photos")
                        gallery = gr.HTML(demo_gallery()[0])
                        next_page = gr.Button("Next photos")
                    with gr.Group(visible=False) as camera_group:
                        gr.HTML(CAPTURE_HTML)
                    demo_index = gr.Textbox("0", elem_id="demo-index", elem_classes="sr-data")
                    demo_submit = gr.Button("Select", elem_id="demo-submit", elem_classes="sr-data")
                    capture_data = gr.Textbox("", elem_id="capture-data", elem_classes="sr-data")
                    capture_submit = gr.Button("Predict", elem_id="capture-submit", elem_classes="sr-data")
                with gr.Column(scale=4, min_width=1000):
                    test_panel = gr.HTML(NO_MODEL)
        with gr.Tab("Inside") as inside_tab:
            with gr.Row():
                with gr.Column(scale=1, min_width=350):
                    filters_panel = gr.HTML(NO_MODEL)
                with gr.Column(scale=1, min_width=350):
                    maps_panel = gr.HTML("")
                with gr.Column(scale=2, min_width=750):
                    pca_panel = gr.HTML("")
    # One GPU queue prevents inference hooks, loading and training from racing.
    gpu_event = dict(concurrency_id="gpu", concurrency_limit=1, api_name=False, show_progress="hidden")
    start.click(train_ui, [model_choice, count, augment], [acc_plot, loss_plot, preview_grid, train_status, confusion_plot], **gpu_event)
    stop.click(stop_training, queue=False, api_name=False)
    load.click(load_last, outputs=[train_status, confusion_plot], **gpu_event)
    source.change(lambda value: (gr.update(visible=value == "Demo photos"), gr.update(visible=value == "Camera/Upload")), source, [demo_group, camera_group], queue=False, api_name=False)
    difficulty.change(lambda f, r: demo_gallery(f, r, 0), [difficulty, reveal], [gallery, page], api_name=False)
    next_page.click(lambda f, r, p: demo_gallery(f, r, p+1), [difficulty, reveal, page], [gallery, page], api_name=False)
    reveal.change(demo_gallery, [difficulty, reveal, page], [gallery, page], api_name=False)
    reveal.change(test_result, [selected, reveal], test_panel, **gpu_event)
    demo_submit.click(select_demo, [demo_index, reveal], [selected, test_panel], **gpu_event)
    capture_submit.click(select_upload, [capture_data, reveal], [selected, test_panel, capture_data], **gpu_event)
    test_tab.select(test_result, [selected, reveal], test_panel, **gpu_event)
    inside_tab.select(inside, selected, [filters_panel, maps_panel, pca_panel], **gpu_event)
app.queue(default_concurrency_limit=1, max_size=16)
print("App ready")
