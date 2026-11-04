"""Audit the actual notebook prep cell without requiring a CUDA runtime.
Usage: python tools/audit_data.py /path/to/kagglecatsanddogs_5340.zip
This developer check does not train a model or launch the teaching app.
"""
import argparse
import json
import os
import tempfile
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('archive', type=Path)
args = parser.parse_args()
archive = args.archive.resolve(strict=True)
repo = Path(__file__).resolve().parents[1]
os.environ['SMOKE_TEST'] = '1'
namespace = {'__name__': '__main__'}
for name in ('00_config.py', '01_setup.py'):
    path = repo / 'notebook_cells' / name
    exec(compile(path.read_text(), str(path), 'exec'), namespace)
with tempfile.TemporaryDirectory(prefix='scout-audit-') as folder:
    root = Path(folder)
    (root / 'petimages.zip').symlink_to(archive)
    namespace.update(SMOKE_TEST=False, ROOT=root, DEMO_ROOT=repo / 'demo')
    path = repo / 'notebook_cells' / '02_data.py'
    exec(compile(path.read_text(), str(path), 'exec'), namespace)
    report = dict(namespace['audit'])
    report.update(environment='Cloud development CPU audit, not Colab',
                  download_included=False, gpu_transfer_included=False,
                  source_archive_bytes=archive.stat().st_size,
                  prepared_shape=list(namespace['DATA'].shape), photos=len(namespace['DEMOS']),
                  workers=os.cpu_count())
    destination = repo / 'artifacts' / 'petimages-exclusion-audit.json'
    destination.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
