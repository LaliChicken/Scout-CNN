"""Build the self-contained Colab notebook from reviewable cell sources."""
from pathlib import Path
import nbformat
ROOT = Path(__file__).resolve().parents[1]
nb = nbformat.v4.new_notebook()
nb.metadata = {"colab": {"name": "ScoutAI.ipynb", "provenance": []}, "accelerator": "GPU",
               "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
               "language_info": {"name": "python"}}
nb.cells.append(nbformat.v4.new_markdown_cell("# Scout AI: Cats vs Dogs\nWe chose ears and whiskers before. Now the network learns its own features.\n\n**Colab GPU → Runtime → Run all.** Then open the share link on the monitor."))
for file in sorted((ROOT / "notebook_cells").glob("*.py")):
    source = file.read_text().rstrip()
    visible = file.name[:2] in {"03", "04", "05"}
    if visible:
        limit = {"03": 25, "04": 15, "05": 20}[file.name[:2]]
        assert len(source.splitlines()) <= limit, (file.name, len(source.splitlines()))
    cell = nbformat.v4.new_code_cell(source)
    cell.id = file.stem
    cell.metadata = {"id": file.stem}
    if not visible:
        cell.metadata.update(cellView="form", jupyter={"source_hidden": True})
    nb.cells.append(cell)
nbformat.validate(nb)
nbformat.write(nb, ROOT / "ScoutAI.ipynb")
print("Built ScoutAI.ipynb")
