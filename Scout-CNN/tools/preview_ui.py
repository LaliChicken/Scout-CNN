"""Developer-only browser fixture. Synthetic CPU data; never a teaching runtime."""
import os
from pathlib import Path
os.environ["SMOKE_TEST"] = "1"
root = Path(__file__).resolve().parents[1]
os.chdir(root)
namespace = {"__name__": "__main__"}
for path in sorted((root / "notebook_cells").glob("*.py")):
    if path.name[:2] <= "06":
        exec(compile(path.read_text(), str(path), "exec"), namespace)
namespace["app"].launch(server_name="127.0.0.1", server_port=7860, share=False,
                        show_api=False, show_error=True, inbrowser=False)
