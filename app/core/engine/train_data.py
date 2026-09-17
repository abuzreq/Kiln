"""Training data for record-based datasets: frame and augment as images load.

A record-based dataset is a list of image paths plus a recipe (see
``app.backend.data.manifest``). When the dataset is built -- once, as training
starts -- the recipe's seed decides a fixed set of variations of every image
(``augment.plan``), so the dataset is ``images x augment_variants`` long and two
runs of the same recipe and seed train on exactly the same thing. Each item is
then, as it loads:

    load RGB -> frame to the recipe's width x height -> apply its variation
    -> fit to the training image size -> tensor -> the engine's normalisation

Both engines use it -- xurdif from its subprocess (via ``--manifest``), Diffusers
in-process -- and each keeps its own fit and value range, so a record dataset
trains the way a folder of the same framed images would, minus the copies.

Nothing is written to disk: a variation is a few numbers, and applying it costs
about a millisecond. Framing is the expensive part, so framed images are kept in
memory up to a byte budget and every variation of an image reuses one framing.
"""
import json
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

    Its length is the number of images times the recipe's variations per image,
    and each item is one (image, variation) pair, so a pass sees every variation
    of every image exactly once.

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
        # The recipe's seed is the dataset's, so both engines build the same
        # set; ``seed`` is only a fallback for a recipe written before seeds.
        self.seed = int(recipe.get("augment_seed", seed if seed is not None else 0))
        self.variants = max(1, int(recipe.get("augment_variants", 1)))
        self.plan = [
            augment.plan(recipe["augmentations"], recipe["augment_settings"],
                         self.variants, self.seed, i)
            for i in range(len(self.files))
        ]
        # An image whose augmentations cannot make that many distinct
        # variations gets fewer, so index by the flat list of (image, variation).
        self.items = [(i, v) for i, ps in enumerate(self.plan) for v in range(len(ps))]
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
        return len(self.items)

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

    def sample(self, j: int):
        """Training item ``j`` as a PIL image (before fit and tensor)."""
        i, v = self.items[j]
        return augment.apply(self.framed(i), self.plan[i][v])

    def __getitem__(self, j):
        return self.tf(self.sample(j))
