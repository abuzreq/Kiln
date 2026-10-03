"""Print SAMPLES entries for app/core/sample_models.py from a folder of .pt files.

    python scripts/sample_models_manifest.py models/pretrained

Run it on exactly the files attached to the release, then paste the output over
SAMPLES and bump RELEASE_TAG. Reads each checkpoint's metadata the way Kiln
does, so a file Kiln cannot load fails here rather than on a user's machine.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.model_manager import read_meta  # noqa: E402
from app.core.sample_models import sha256_of  # noqa: E402


def main(folder: str) -> None:
    files = sorted(Path(folder).glob("*.pt"))
    if not files:
        sys.exit(f"no .pt files in {folder}")
    print("SAMPLES = [")
    for p in files:
        meta = read_meta(p)
        print(f'    {{"file": "{p.name}", "size": {p.stat().st_size}, '
              f'"mtype": "{meta.mtype}", "step": {meta.step},')
        print(f'     "sha256": "{sha256_of(p)}"}},')
    print("]")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "models/pretrained")
