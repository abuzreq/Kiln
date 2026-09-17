"""Training data for record-based datasets: frame and augment as images load.

A record-based dataset is a list of image paths plus a recipe (see
``app.backend.data.manifest``). Nothing is written ahead of time; each time the
trainer asks for an image this does:

    load RGB -> frame to the recipe's width x height -> one random augmentation
    -> fit to the training image size -> tensor -> the engine's normalisation

Both engines use it -- xurdif from its subprocess (via ``--manifest``), Diffusers
in-process -- and each keeps its own fit and value range, so a record dataset
trains the way a folder of the same framed images would, minus the copies.

Framing is deterministic and can be expensive on large originals, so framed
images are kept in memory up to a byte budget; the random augmentation is
cheap and drawn fresh every time.
"""
import json
import random
import sys
from collections import OrderedDict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:  # the xurdif subprocess runs from vendor/xurdif
    sys.path.insert(0, str(ROOT))

from app.backend.data import augment, framing  # noqa: E402

# Framed images kept in RAM: 2 GiB is ~2,700 images at 512x512.
CACHE_BYTES = 2 << 30


def read_snapshot(path: str | Path) -> dict:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not data.get("files"):
        raise ValueError(f"dataset snapshot {path} lists no images")
    return data


class ManifestDataset:
    """A torch-style dataset over a snapshot's files and recipe.

    ``fit`` and ``normalize`` are the engine's: xurdif resizes to a square (or
    random-crops) and shifts to [-0.5, 0.5]; Diffusers resizes the short side
    before a random crop and normalises to [-1, 1].
    """

    def __init__(self, files, recipe: dict, image_size: int, fit: str = "resize",
                 engine: str = "xurdif", cache_bytes: int = CACHE_BYTES, seed: int | None = None):
        from torchvision import transforms

        self.files = [str(f) for f in files]
        if not self.files:
            raise ValueError("no images in this dataset")
        self.recipe = recipe
        self.rng = random.Random(seed)
        self._cache: "OrderedDict[int, object]" = OrderedDict()
        self._cache_used = 0
        self._cache_bytes = cache_bytes

        if engine == "xurdif":
            geom = ([transforms.Resize((image_size, image_size))] if fit == "resize"
                    else [transforms.RandomCrop((image_size, image_size), pad_if_needed=True,
                                                padding_mode="reflect")])
            tail = [transforms.ToTensor(), transforms.Lambda(lambda t: t - 0.5)]
        else:
            geom = ([transforms.Resize((image_size, image_size))] if fit == "resize"
                    else [transforms.Resize(image_size), transforms.RandomCrop(image_size)])
            tail = [transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
        self.tf = transforms.Compose(geom + tail)

    @classmethod
    def from_snapshot(cls, path, image_size: int, fit: str = "resize", engine: str = "xurdif", **kw):
        snap = read_snapshot(path)
        return cls(snap["files"], snap["recipe"], image_size, fit, engine, **kw)

    def __len__(self):
        return len(self.files)

    def framed(self, i: int):
        """The i-th image framed to the recipe, from memory when possible."""
        from PIL import Image

        hit = self._cache.get(i)
        if hit is not None:
            self._cache.move_to_end(i)
            return hit
        r = self.recipe
        with Image.open(self.files[i]) as im:
            img = framing.process_image(im, r["width"], r["height"], r["resize_mode"], r["padding_mode"])
        size = img.width * img.height * 3
        if size <= self._cache_bytes:
            self._cache[i] = img
            self._cache_used += size
            while self._cache_used > self._cache_bytes and self._cache:
                _, old = self._cache.popitem(last=False)
                self._cache_used -= old.width * old.height * 3
        return img

    def sample(self, i: int):
        """One training view of the i-th image, as a PIL image (before fit/tensor)."""
        r = self.recipe
        return augment.random_augment(self.framed(i), r["augmentations"], r["augment_settings"], self.rng)

    def __getitem__(self, i):
        return self.tf(self.sample(i))
