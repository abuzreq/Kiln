"""The sample models: xurdif checkpoints Kiln can fetch in one click.

Trained by Hannu Töyrylä, who wrote xurdif, and made freely available by him
with no restrictions on their use. They are not in the repository -- weights in
git history would ride along with every clone forever -- but attached to a
GitHub release of Kiln, so they come from the same place as the code and nothing
has to be copied by hand.

Each entry pins the file's size and SHA-256. A download that does not match is
discarded rather than kept: a truncated or substituted checkpoint would
otherwise be loaded with ``torch.load``.

To publish a new set, attach the files to a release, then update ``RELEASE_TAG``
and the entries below (``scripts/sample_models_manifest.py`` prints them).
"""
from __future__ import annotations

import hashlib
from pathlib import Path

REPO = "abuzreq/kiln"
RELEASE_TAG = "sample-models-v1"
RELEASE_PAGE = f"https://github.com/{REPO}/releases/tag/{RELEASE_TAG}"
AUTHOR = "Hannu Töyrylä"

SAMPLES = [
    {"file": "outshn-145.pt", "size": 85326330, "mtype": "tinyunet_conf_attention", "step": 14500,
     "sha256": "55054dce18044c66f35a2e40ba006748955dfc4435074d4143b73be8da1d33fb"},
    {"file": "outshn-48.pt", "size": 85326136, "mtype": "tinyunet_conf_attention", "step": 4800,
     "sha256": "50cf74e3b2c5783fad7eddbc29dd582f7abc53eace70fe233ae9410f3a82c19a"},
    {"file": "tnxmgl-58.pt", "size": 30232669, "mtype": "tinyunet_with_attention3", "step": 5800,
     "sha256": "23ede4275fd910fe54f3412abc86b31bb1f58713134fecbbb2a354817fa527ce"},
    {"file": "tnxsyn-28.pt", "size": 42418470, "mtype": "tinyunet_with_attention3", "step": 2800,
     "sha256": "ef7a2f4dad36fef3c3ad87c07673a1c19d3058cd96ad0d11915ffaf61b25a8f3"},
    {"file": "txcrcl-10.pt", "size": 30232669, "mtype": "tinyunet_with_attention3", "step": 1000,
     "sha256": "69771e6c689f8f647e59fb9a5f05e9509b58ab95ca6daced6153539384bf7773"},
    {"file": "txkrs0b-114.pt", "size": 30232843, "mtype": "tinyunet_with_attention3", "step": 11400,
     "sha256": "f5eaae069f1dfaea4ec5564399ea8124668cc3ffa0fa3b18ab5711724fbd47f2"},
    {"file": "txplsa-200.pt", "size": 30232843, "mtype": "tinyunet_with_attention3", "step": 20000,
     "sha256": "b5cd3dad08f89632a089bed8e46cf16482ea0c7356b6c0560d0f79a1942b5a9d"},
]


def url_for(entry: dict) -> str:
    return f"https://github.com/{REPO}/releases/download/{RELEASE_TAG}/{entry['file']}"


def by_file(name: str) -> dict | None:
    return next((s for s in SAMPLES if s["file"] == name), None)


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def installed_where(entry: dict, folders: list[Path]) -> Path | None:
    """Where this sample already is, if anywhere Kiln looks.

    Matched by name and size, not hash: hashing 300 MB on every listing would
    make opening the Models tab slow, and a same-named file of the same size is
    the sample in every case that matters. Downloads are hashed in full.
    """
    for d in folders:
        p = d / entry["file"]
        try:
            if p.is_file() and p.stat().st_size == entry["size"]:
                return p
        except OSError:
            continue
    return None
