#@title 2 · Data { display-mode: "form" }
import multiprocessing as mp, importlib.util, zipfile, urllib.request, shutil
from sklearn.model_selection import train_test_split
PREP_STARTED = time.perf_counter()
# A real module makes Pool workers importable in a notebook and in nbconvert.
WORKER_SOURCE = r'''
from pathlib import Path
import hashlib, warnings
import numpy as np
from PIL import Image, ImageOps

def average_hash(image):
    a = np.asarray(image.convert("L").resize((8, 8), Image.Resampling.LANCZOS))
    return int.from_bytes(np.packbits(a > a.mean()).tobytes(), "big")

def canonical(image):
    image = image.convert("RGB")
    w, h = image.size
    scale = 144 / min(w, h)
    image = image.resize((max(144, round(w*scale)), max(144, round(h*scale))), Image.Resampling.BILINEAR)
    return np.asarray(ImageOps.fit(image, (144, 144), method=Image.Resampling.BILINEAR)).copy()

def init_worker(exact, near):
    global DEMO_SHA, DEMO_HASH
    DEMO_SHA, DEMO_HASH = exact, near

def prep_file(item):
    path, label = item
    try:
        raw = Path(path).read_bytes()
        if not raw:
            return ("invalid", label, None, None, None)
        sha = hashlib.sha1(raw).hexdigest()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with Image.open(path) as opened:
                rgb = opened.convert("RGB")
                ah = average_hash(rgb)
                if sha in DEMO_SHA:
                    return ("exact", label, None, sha, ah)
                if any((ah ^ dh).bit_count() <= 5 for dh in DEMO_HASH):
                    return ("near", label, None, sha, ah)
                return ("ok", label, canonical(rgb), sha, ah)
    except (OSError, ValueError, SyntaxError, Image.DecompressionBombError):
        return ("invalid", label, None, None, None)
'''
worker_path = ROOT / "scout_prep.py"
worker_path.write_text(WORKER_SOURCE)
sys.path.insert(0, str(ROOT)) if str(ROOT) not in sys.path else None
import scout_prep
importlib.reload(scout_prep)

def safe_extract(archive, destination):
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        for item in z.infolist():
            target = (destination / item.filename).resolve()
            if not target.is_relative_to(destination.resolve()):
                raise ValueError("Unsafe archive member")
        z.extractall(destination)

def fetch_petimages():
    destination = ROOT / "source"
    marker = destination / ".complete"
    if marker.exists():
        return destination
    archive = ROOT / "petimages.zip"
    try:
        if not archive.exists():
            temporary = archive.with_suffix(".part")
            with urllib.request.urlopen(DATA_URL, timeout=90) as response, temporary.open("wb") as out:
                total = int(response.headers.get("Content-Length", 0))
                with tqdm(total=total, unit="B", unit_scale=True, desc="Download") as bar:
                    while chunk := response.read(4 * 1024 * 1024):
                        out.write(chunk); bar.update(len(chunk))
            temporary.replace(archive)
        safe_extract(archive, destination)
    except Exception as error:
        print(f"Download unavailable ({type(error).__name__}); checking mounted Drive.")
        drive = Path("/content/drive/MyDrive")
        archives = {name: next(drive.rglob(name), None) if drive.exists() else None for name in ("Cat.zip", "Dog.zip")}
        if not all(archives.values()):
            raise RuntimeError("Download failed. Fallback needs Cat.zip and Dog.zip in an already mounted Drive folder.") from error
        # Separate roots handle both flat archives and archives with a Cat/Dog folder.
        for name, source in archives.items():
            safe_extract(source, destination / "PetImages" / Path(name).stem)
    marker.touch()
    return destination

DEMOS = []
if SMOKE_TEST:
    rng = np.random.default_rng(SEED)
    pixels = rng.integers(0, 256, (80, 144, 144, 3), dtype=np.uint8)
    labels = np.tile([0, 1], 40).astype(np.int64)
    for i in range(50):
        DEMOS.append({"image": pixels[i].copy(), "label": i % 2, "difficulty": "tricky" if i < 16 else "easy", "id": i})
    audit = {"synthetic": True, "invalid": 0, "exact": 0, "near": 0, "demo_remaining": 0}
else:
    demo_files = sorted(DEMO_ROOT.glob("*/*/*.jpg"))
    assert len(demo_files) == 50, f"Expected 50 held-out JPGs; found {len(demo_files)}."
    demo_sha, demo_hash = set(), []
    for i, path in enumerate(demo_files):
        assert path.parent.name in ("easy", "tricky") and path.parent.parent.name in ("cats", "dogs")
        with Image.open(path) as opened:
            rgb = opened.convert("RGB")
            demo_sha.add(hashlib.sha1(path.read_bytes()).hexdigest())
            demo_hash.append(scout_prep.average_hash(rgb))
            DEMOS.append({"image": scout_prep.canonical(rgb), "label": int(path.parent.parent.name == "dogs"), "difficulty": path.parent.name, "id": i})
    assert sum(d["difficulty"] == "tricky" for d in DEMOS) == 16, "Expected 16 tricky photos."
    source = fetch_petimages()
    items = []
    for label, name in enumerate(("Cat", "Dog")):
        folder = next(p for p in source.rglob(name) if p.is_dir())
        items.extend((str(p), label) for p in sorted(folder.rglob("*")) if p.is_file())
    # Allocate one uint8 RAM buffer; keep indices rather than copied train/val arrays.
    pixels = np.empty((min(len(items), max_per_class * 2), 144, 144, 3), dtype=np.uint8)
    labels = np.empty(len(pixels), dtype=np.int64)
    kept_sha, kept_hash, counts, used = [], [], [0, 0], 0
    audit = {"synthetic": False, "invalid": 0, "exact": 0, "near": 0, "capped": 0}
    with mp.get_context("fork").Pool(os.cpu_count(), initializer=scout_prep.init_worker, initargs=(demo_sha, demo_hash)) as pool:
        for status, label, array, sha, ah in tqdm(pool.imap(scout_prep.prep_file, items, chunksize=32), total=len(items), desc="Prepare"):
            if status != "ok":
                audit[status] += 1
            elif counts[label] < max_per_class:
                pixels[used], labels[used] = array, label
                counts[label] += 1; used += 1
                kept_sha.append(sha); kept_hash.append(ah)
            else:
                audit["capped"] += 1
    pixels, labels = pixels[:used], labels[:used]
    audit["demo_remaining"] = sum(sha in demo_sha or any((ah ^ dh).bit_count() <= 5 for dh in demo_hash) for sha, ah in zip(kept_sha, kept_hash))
    assert audit["demo_remaining"] == 0, "Held-out image leakage detected."
    assert min(counts) >= 250, "Not enough valid images per class."
train_ids, val_ids = train_test_split(np.arange(len(labels)), test_size=0.1, stratify=labels, random_state=SEED)
# GPU holds the whole dataset, once. No DataLoader or per-batch host transfers.
DATA = torch.from_numpy(pixels).permute(0, 3, 1, 2).to(DEVICE).contiguous(memory_format=torch.channels_last)
LABELS = torch.from_numpy(labels).to(DEVICE)
TRAIN_IDS = torch.tensor(train_ids, device=DEVICE)
VAL_IDS = torch.tensor(val_ids, device=DEVICE)
PREVIEW_IDS = VAL_IDS[:8]
PREP_SECONDS = time.perf_counter() - PREP_STARTED
audit.update(prep_s=PREP_SECONDS, train=len(train_ids), val=len(val_ids))
(ROOT / "prep_report.json").write_text(json.dumps(audit, indent=2))
print(f"Skipped invalid: {audit['invalid']} · Removed exact: {audit['exact']} · near: {audit['near']}")
print(f"Demo photos left in training: {audit['demo_remaining']} · prep: {PREP_SECONDS:.1f}s")
fig, axes = plt.subplots(4, 8, figsize=(16, 8))
for ax, i in zip(axes.flat, np.random.default_rng(SEED).choice(train_ids, 32, replace=False)):
    ax.imshow(pixels[i]); ax.set_title(CLASS_NAMES[labels[i]]); ax.axis("off")
fig.tight_layout(); display(fig); plt.close(fig)
