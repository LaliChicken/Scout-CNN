#@title 1 · Setup { display-mode: "form" }
import sys, subprocess, time, json, math, random, hashlib, io, base64, html, threading
from pathlib import Path
if not SMOKE_TEST:
    assert "google.colab" in sys.modules or __import__("importlib").util.find_spec("google.colab"), "Use a Colab GPU runtime."
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "gradio==5.49.1"], check=True)
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from PIL import Image, ImageOps
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from IPython.display import display, HTML
from tqdm.auto import tqdm
import gradio as gr
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if SMOKE_TEST:
    DEVICE = torch.device("cpu")
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    ROOT = Path.cwd() / ".smoke"
else:
    assert torch.cuda.is_available(), "Choose Runtime → Change runtime type → GPU."
    DEVICE = torch.device("cuda")
    gpu = torch.cuda.get_device_properties(0)
    print(f"GPU: {gpu.name} · {gpu.total_memory / 2**30:.1f} GiB")
    torch.backends.cudnn.benchmark = True
    ROOT = Path("/content/scout-ai")
ROOT.mkdir(parents=True, exist_ok=True)
CHECKPOINTS = (ROOT / "checkpoints") if SMOKE_TEST else Path("/content/checkpoints")
CHECKPOINTS.mkdir(parents=True, exist_ok=True)
if not SMOKE_TEST:
    assert "/OWNER/" not in REPO_URL, "Set REPO_URL to the GitHub repository containing this notebook and demo photos."
    REPO = ROOT / "repo"
    if not (REPO / ".git").exists():
        subprocess.run(["git", "clone", "--depth", "1", REPO_URL, str(REPO)], check=True)
    else:
        subprocess.run(["git", "-C", str(REPO), "pull", "--ff-only"], check=True)
    DEMO_ROOT = REPO / "demo"
AMP_DTYPE = torch.bfloat16 if DEVICE.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float16
AMP_ENABLED = DEVICE.type == "cuda"
CLASS_NAMES = ["CAT", "DOG"]
print("Setup ready" + (" · synthetic CPU smoke test" if SMOKE_TEST else ""))
