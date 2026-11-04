#@title 0 · Config { display-mode: "form" }
import os
DATA_URL = "https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_5340.zip" #@param {type:"string"}
epochs = 8 #@param {type:"integer"}
img_size = 128 #@param [96, 128, 144] {type:"raw"}
max_per_class = 12500 #@param {type:"integer"}
SHARE_PASSWORD = "" #@param {type:"string"}
SMOKE_TEST = False #@param {type:"boolean"}
SMOKE_TEST = SMOKE_TEST or os.environ.get("SMOKE_TEST") == "1"
REPO_URL = "https://github.com/LaliChicken/Scout-CNN.git"  # Repository containing the held-out photos.
SEED = 42
assert epochs >= 1 and 32 <= img_size <= 144 and max_per_class >= 280
