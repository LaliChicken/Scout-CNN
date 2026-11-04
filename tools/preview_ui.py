"""Developer browser fixture: synthetic training data, real held-out demo photos."""
import os
from pathlib import Path
os.environ["SMOKE_TEST"] = "1"
root = Path(__file__).resolve().parents[1]
os.chdir(root)
namespace = {"__name__": "__main__"}
for path in sorted((root / "notebook_cells").glob("*.py")):
    if path.name[:2] <= "06":
        exec(compile(path.read_text(), str(path), "exec"), namespace)
        if path.name.startswith("02"):
            # Exercise the shipped gallery without adding demo photos to training.
            demo_files = sorted((root / "demo").glob("*/*/*.jpg"))
            if demo_files:
                assert len(demo_files) == 50
                demos = []
                for i, photo in enumerate(demo_files):
                    with namespace["Image"].open(photo) as opened:
                        array = namespace["scout_prep"].canonical(opened.convert("RGB"))
                    demos.append({"image": array, "label": int(photo.parent.parent.name == "dogs"),
                                  "difficulty": photo.parent.name, "id": i})
                namespace["DEMOS"] = demos
namespace["app"].launch(server_name="127.0.0.1", server_port=7860, share=False,
                        show_api=False, show_error=True, inbrowser=False)
