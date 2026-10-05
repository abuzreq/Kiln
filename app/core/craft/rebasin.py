"""Re-basin: reorder one model's units so they line up with another's.

Two models trained apart learn similar features in different channel orders --
hidden units are interchangeable, so each run settles on its own permutation.
Averaging them pairs unrelated features. Git Re-Basin (Ainsworth, Hayase and
Srinivasa, ICLR 2023) finds, for every layer, the permutation of B's units that
best matches A's, and applies it everywhere that layer's units are read, so
B computes exactly the same function in A's order. Only then does a blend mean
"this feature of A with the same feature of B".

A *spec* lists the permutation groups of one architecture. A group is one set
of interchangeable units -- a conv's output channels, say -- and names every
place they appear: the conv's own weight rows and bias, the FiLM rows that
scale and shift them, and the input columns of whatever reads them next. Each
place is a segment ``(key, axis, offset, size)``; a FiLM projection's gamma and
beta halves are two segments of the same group, and a decoder conv's input,
[upsampled | skip], is two segments of two different groups.

Two ways to find the permutations, both from the paper:

- **Weight matching** (algorithm 1) needs no inputs: coordinate descent over the
  groups, each step a linear assignment that maximises the summed inner product
  of A's and B's weights in every segment, with every other group's current
  permutation applied. On Kiln's narrow UNets trained apart it only finds
  chance correlations -- it scores the same on two untrained networks -- and
  made 50% merges worse in tests, so merges do not offer it. It stays as the
  spec's check: a shuffled copy of a model is matched back exactly.
- **Activation matching** correlates each unit's response in A with every
  unit's in B, on the same inputs, and solves one assignment per group. The
  inputs can be noised samples of A's and B's own outputs (``noised_inputs``),
  so no dataset is needed. It was the only variant that changed what a merge of
  unrelated models looks like.

Both are deterministic, so a merge that aligns B saves and replays the same way.

Specs exist for the xurdif ``tinyunet_with_attention3`` architecture, whose
channel LayerNorm has no parameters and averages over channels, so it commutes
with any permutation. Diffusers' GroupNorm does not, which is why Diffusers
models are out of scope here.
"""
from collections import OrderedDict
from dataclasses import dataclass, field

ATTN3 = "tinyunet_with_attention3"


@dataclass
class Segment:
    key: str
    axis: int
    offset: int
    size: int


@dataclass
class Spec:
    #: group name -> its segments, in a fixed order (the matching order)
    groups: dict = field(default_factory=dict)
    #: group name -> the module(s) whose output carries its units, for
    #: activation matching; several are concatenated along the samples
    probes: dict = field(default_factory=dict)

    def add(self, group: str, key: str, axis: int, offset: int = 0, size: int | None = None,
            state: dict | None = None):
        n = size if size is not None else state[key].shape[axis] - offset
        self.groups.setdefault(group, []).append(Segment(key, axis, offset, n))

    def size_of(self, group: str) -> int:
        return self.groups[group][0].size

    def by_key(self) -> dict:
        """key -> [(group, segment)], for applying permutations tensor by tensor."""
        out = {}
        for g, segs in self.groups.items():
            for s in segs:
                out.setdefault(s.key, []).append((g, s))
        return out


def supports(mtype: str | None) -> bool:
    return mtype == ATTN3


def spec_for(mtype: str, state: dict, prefix: str = "") -> Spec:
    """The permutation groups of one architecture, sized from ``state`` itself."""
    if mtype != ATTN3:
        raise ValueError(f"no re-basin spec for {mtype}")
    return _attn3_spec(state, prefix)


def _attn3_spec(state: dict, p: str) -> Spec:
    """``tinyunet_with_attention3`` (vendor/xurdif/alt_models/tinyunet_with_attn3.py).

    Forward: init_conv -> per level, ConvBlock (conv, channel LayerNorm, FiLM,
    SiLU), kept as the skip, then a stride-2 conv -> mid_block1 -> attention,
    ``proj(attn(q, k) v + x)`` -> mid_block2 -> per level, a transposed conv,
    concat with the matching skip, ConvBlock -> final_conv. Time: sinusoidal
    embedding -> time_mlp (Linear, SiLU, Linear) -> every FiLM's Linear.
    """
    s = Spec()
    levels = 0
    while f"{p}downs.{levels}.0.conv.weight" in state:
        levels += 1

    def block(group, name):
        """A ConvBlock's output units: conv rows and bias, and both FiLM halves."""
        n = state[f"{p}{name}.conv.weight"].shape[0]
        s.add(group, f"{p}{name}.conv.weight", 0, size=n)
        s.add(group, f"{p}{name}.conv.bias", 0, size=n)
        for half in (0, n):
            s.add(group, f"{p}{name}.film.mlp.1.weight", 0, half, n)
            s.add(group, f"{p}{name}.film.mlp.1.bias", 0, half, n)
        return n

    s.add("time.hidden", f"{p}time_mlp.0.weight", 0, state=state)
    s.add("time.hidden", f"{p}time_mlp.0.bias", 0, state=state)
    s.add("time.hidden", f"{p}time_mlp.2.weight", 1, state=state)
    s.add("time.out", f"{p}time_mlp.2.weight", 0, state=state)
    s.add("time.out", f"{p}time_mlp.2.bias", 0, state=state)
    # ... read by every block's FiLM projection, through a SiLU (elementwise).
    for key in state:
        if key.startswith(p) and key.endswith(".film.mlp.1.weight"):
            s.add("time.out", key, 1, state=state)

    s.add("init", f"{p}init_conv.weight", 0, state=state)
    s.add("init", f"{p}init_conv.bias", 0, state=state)
    s.add("init", f"{p}downs.0.0.conv.weight", 1, state=state)

    skip_into = {}  # down level -> (up conv key, offset of the skip in its input)
    for j in range(levels):
        i = levels - 1 - j
        up_out = state[f"{p}ups.{j}.0.weight"].shape[1]
        skip_into[i] = (f"{p}ups.{j}.1.conv.weight", up_out)

    for i in range(levels):
        g = f"down{i}.block"
        n = block(g, f"downs.{i}.0")
        s.add(g, f"{p}downs.{i}.1.weight", 1, size=n)
        key, off = skip_into[i]
        s.add(g, key, 1, off, n)
        g = f"down{i}.down"
        s.add(g, f"{p}downs.{i}.1.weight", 0, state=state)
        s.add(g, f"{p}downs.{i}.1.bias", 0, state=state)
        nxt = f"{p}downs.{i + 1}.0.conv.weight" if i + 1 < levels else f"{p}mid_block1.conv.weight"
        s.add(g, nxt, 1, state=state)

    # mid_block1's units are also the attention's residual stream: v must keep
    # them, because the block adds v's output straight back onto its input.
    n = block("mid1", "mid_block1")
    for k in ("q", "k", "v"):
        s.add("mid1", f"{p}mid_attn.{k}.weight", 1, size=n)
    s.add("mid1", f"{p}mid_attn.v.weight", 0, size=n)
    s.add("mid1", f"{p}mid_attn.v.bias", 0, size=n)
    s.add("mid1", f"{p}mid_attn.proj.weight", 1, size=n)
    # q and k only meet in q . k, so they share one free order of their own.
    for k in ("q", "k"):
        s.add("attn.qk", f"{p}mid_attn.{k}.weight", 0, state=state)
        s.add("attn.qk", f"{p}mid_attn.{k}.bias", 0, state=state)
    s.add("attn.proj", f"{p}mid_attn.proj.weight", 0, state=state)
    s.add("attn.proj", f"{p}mid_attn.proj.bias", 0, state=state)
    s.add("attn.proj", f"{p}mid_block2.conv.weight", 1, state=state)
    n = block("mid2", "mid_block2")
    s.add("mid2", f"{p}ups.0.0.weight", 0, size=n)   # ConvTranspose2d: (in, out, kh, kw)

    for j in range(levels):
        g = f"up{j}.up"
        n = state[f"{p}ups.{j}.0.weight"].shape[1]
        s.add(g, f"{p}ups.{j}.0.weight", 1, size=n)
        s.add(g, f"{p}ups.{j}.0.bias", 0, size=n)
        s.add(g, f"{p}ups.{j}.1.conv.weight", 1, 0, n)
        g = f"up{j}.block"
        n = block(g, f"ups.{j}.1")
        nxt = (f"{p}ups.{j + 1}.0.weight", 0) if j + 1 < levels else (f"{p}final_conv.weight", 1)
        s.add(g, nxt[0], nxt[1], size=n)

    s.probes = {"time.hidden": ["time_mlp.0"], "time.out": ["time_mlp.2"], "init": ["init_conv"],
                "mid1": ["mid_block1"], "attn.qk": ["mid_attn.q", "mid_attn.k"],
                "attn.proj": ["mid_attn.proj"], "mid2": ["mid_block2"]}
    for i in range(levels):
        s.probes[f"down{i}.block"] = [f"downs.{i}.0"]
        s.probes[f"down{i}.down"] = [f"downs.{i}.1"]
    for j in range(levels):
        s.probes[f"up{j}.up"] = [f"ups.{j}.0"]
        s.probes[f"up{j}.block"] = [f"ups.{j}.1"]
    if set(s.probes) != set(s.groups):
        raise ValueError(f"re-basin spec: probes {sorted(set(s.probes) ^ set(s.groups))} unmatched")

    for g, segs in s.groups.items():
        sizes = {seg.size for seg in segs}
        if len(sizes) != 1:
            raise ValueError(f"re-basin spec: group {g} has segments of sizes {sorted(sizes)}")
    return s


def identity(spec: Spec) -> dict:
    import torch

    return {g: torch.arange(spec.size_of(g)) for g in spec.groups}


def apply(state: dict, spec: Spec, perms: dict, skip: tuple | None = None) -> dict:
    """``state`` with each group's permutation applied wherever its units appear.

    ``perms[g][i]`` is the unit of the input that lands at position ``i``. With
    ``skip = (key, axis)``, that one axis of that one tensor is left as it is
    (weight matching reads a tensor with every axis but its own permuted).
    """
    import torch

    out = dict(state)
    for key, segs in spec.by_key().items():
        t = state[key]
        axes = {}
        for g, seg in segs:
            if skip == (key, seg.axis):
                continue
            idx = axes.setdefault(seg.axis, torch.arange(t.shape[seg.axis]))
            idx[seg.offset:seg.offset + seg.size] = seg.offset + perms[g]
        for axis, idx in axes.items():
            t = t.index_select(axis, idx)
        out[key] = t
    return out


def random_perms(spec: Spec, seed: int = 0) -> dict:
    import torch

    g = torch.Generator().manual_seed(seed)
    return {name: torch.randperm(spec.size_of(name), generator=g) for name in spec.groups}


def match(state_a: dict, state_b: dict, spec: Spec, max_sweeps: int = 50) -> dict:
    """Permutations of B's units that best line them up with A's (weight matching).

    Returns ``perms`` for ``apply(state_b, spec, perms)``. Each step solves one
    group as a linear assignment over the summed inner products of its
    segments, with every other group's current order applied; sweeps repeat
    until a whole sweep changes nothing.
    """
    import torch
    from scipy.optimize import linear_sum_assignment

    perms = identity(spec)
    keys = spec.by_key()
    a64 = {k: state_a[k].double() for k in keys}
    b64 = {k: state_b[k].double() for k in keys}
    for _ in range(max_sweeps):
        changed = False
        for g, segs in spec.groups.items():
            n = spec.size_of(g)
            cost = torch.zeros(n, n, dtype=torch.float64)
            for seg in segs:
                wa = a64[seg.key].narrow(seg.axis, seg.offset, n)
                wb = apply({seg.key: b64[seg.key]}, _only(spec, seg.key), perms,
                           skip=(seg.key, seg.axis))[seg.key].narrow(seg.axis, seg.offset, n)
                cost += wa.movedim(seg.axis, 0).reshape(n, -1) @ wb.movedim(seg.axis, 0).reshape(n, -1).T
            _, cols = linear_sum_assignment(cost.numpy(), maximize=True)
            new = torch.as_tensor(cols, dtype=torch.long)
            if not torch.equal(new, perms[g]):
                perms[g] = new
                changed = True
        if not changed:
            break
    return perms


def _only(spec: Spec, key: str) -> Spec:
    """The part of ``spec`` that touches one tensor, so applying it costs one tensor."""
    sub = Spec()
    for g, segs in spec.groups.items():
        for seg in segs:
            if seg.key == key:
                sub.groups.setdefault(g, []).append(seg)
    return sub


# --- activation matching ------------------------------------------------------

def noised_inputs(x0, alphas_cumprod, timesteps=(20, 100, 250, 400, 550, 700, 850, 970),
                  seed: int = 0):
    """Probe inputs: ``x0`` noised to each timestep, as the network sees it while sampling.

    ``x0`` is best a few samples of A's and B's own outputs -- the inputs these
    models actually meet -- so matching needs no dataset.
    """
    import torch

    g = torch.Generator(device=x0.device).manual_seed(seed)
    xs, ts = [], []
    for t in timesteps:
        a = alphas_cumprod[t].to(x0.device)
        xs.append(a.sqrt() * x0 + (1 - a).sqrt() * torch.randn(x0.shape, generator=g, device=x0.device))
        ts.append(torch.full((x0.shape[0],), float(t), device=x0.device))
    return torch.cat(xs), torch.cat(ts)


def activations(net, spec: Spec, x, t, chunk: int = 16, keep: int = 8192) -> dict:
    """Each group's units as rows, a sample of input positions as columns.

    Run in chunks, each keeping an evenly spaced share of its positions and
    moving it to the CPU at once: a full record of every probed layer for a
    few hundred inputs runs to gigabytes, and on a card shared with anything
    else that spills into system memory and crawls. Correlations need a few
    thousand samples per unit, not millions.
    """
    import torch

    mods = dict(getattr(net, "wrapped", net).named_modules())
    names = {n for ns in spec.probes.values() for n in ns}
    chunks = max(1, -(-len(x) // chunk))
    per = max(1, keep // chunks)
    out = {g: [] for g in spec.probes}

    def rows(o):
        return o.transpose(0, 1).reshape(o.shape[1], -1) if o.dim() == 4 else o.T

    for i in range(0, len(x), chunk):
        seen = {}
        hooks = [mods[n].register_forward_hook(
            lambda m, inp, o, n=n: seen.__setitem__(n, o.detach())) for n in names]
        try:
            with torch.no_grad():
                net(x[i:i + chunk], t[i:i + chunk])
        finally:
            for h in hooks:
                h.remove()
        for g, ns in spec.probes.items():
            r = torch.cat([rows(seen[n]) for n in ns], 1)
            cols = torch.linspace(0, r.shape[1] - 1, min(per, r.shape[1]), device=r.device).long()
            out[g].append(r[:, cols].float().cpu())
        del seen
    return {g: torch.cat(parts, 1) for g, parts in out.items()}


def match_activations(acts_a: dict, acts_b: dict) -> dict:
    """Permutations of B's units that best correlate with A's, group by group.

    ``acts_*`` come from ``activations`` on the same inputs. Each unit is
    standardised over the inputs; one linear assignment per group maximises the
    summed correlation of matched pairs.
    """
    import torch
    from scipy.optimize import linear_sum_assignment

    perms = {}
    for g, a in acts_a.items():
        b = acts_b[g]
        a = (a - a.mean(1, keepdim=True)) / a.std(1, keepdim=True).clamp_min(1e-8)
        b = (b - b.mean(1, keepdim=True)) / b.std(1, keepdim=True).clamp_min(1e-8)
        corr = (a.double() @ b.double().T / a.shape[1]).cpu().numpy()
        _, cols = linear_sum_assignment(corr, maximize=True)
        perms[g] = torch.as_tensor(cols, dtype=torch.long)
    return perms


# --- aligning one model to another, for merges ---------------------------------
#: "Line up B with A" in a merge: off, or by activation matching.
ALIGN = ("none", "activations")
#: Activation matching probes with the models' own samples: this many of each,
#: small and short, on a fixed seed, so the probe never depends on the
#: sampling settings of the moment.
PROBE_SAMPLES, PROBE_SIZE, PROBE_STEPS = 8, 64, 20

_PERMS: "OrderedDict[tuple, tuple]" = OrderedDict()
_PERMS_MAX = 16


def can_align(path: str) -> bool:
    from app.core.model_manager import read_meta

    return supports(read_meta(path).mtype)


def _prefix_of(state: dict) -> str:
    from app.core.backends.xurdif.loader import DENOISE_PREFIX

    return DENOISE_PREFIX if any(k.startswith(DENOISE_PREFIX) for k in state) else ""


class Cancelled(Exception):
    """An alignment stopped because its job was cancelled; nothing was kept."""


def pair_perms(path_a: str, path_b: str, how: str, device: str = "cpu", ema: bool = True,
               cancel=None):
    """``(spec, perms)`` that reorder ``path_b``'s units to line up with ``path_a``'s.

    Found on the slot the sampler reads and applied to every slot: a model's
    raw and averaged weights share one unit order. Kept per file state and
    method, so a ladder and the merge saved from it use the same permutation --
    activation matching samples on the GPU, which need not repeat bit for bit.
    ``cancel`` (a callable) stops activation matching's probe; it raises
    ``Cancelled`` and nothing is kept.
    """
    from app.core import backends, library
    from utils.exceptions import ValidationError

    if how not in ALIGN or how == "none":
        raise ValidationError(f"unknown alignment: {how}")

    def stamp(p):
        f = library.weights_file(p)
        return (str(p), *(library.file_stamp(f) if f else ()))

    key = (stamp(path_a), stamp(path_b), how, bool(ema))
    if key in _PERMS:
        _PERMS.move_to_end(key)
        return _PERMS[key]
    backend, ref_a = backends.resolve(path_a)
    _, ref_b = backends.resolve(path_b)
    meta = backend.describe(ref_a)
    if not supports(meta.mtype):
        raise ValidationError(f"aligning is not available for {meta.mtype} models yet")
    slots_a, slots_b = backend.merge_slots(ref_a), backend.merge_slots(ref_b)
    slot = backend.sampled_slot(slots_a, ema)
    sa = backend.net_state(slots_a[slot])
    sb = backend.net_state(slots_b.get(slot) or next(iter(slots_b.values())))
    spec = spec_for(meta.mtype, sa, _prefix_of(sa))
    perms = _activation_perms(backend, ref_a, path_a, sa, sb, spec, device, ema, cancel)
    _PERMS[key] = (spec, perms)
    while len(_PERMS) > _PERMS_MAX:
        _PERMS.popitem(last=False)
    return spec, perms


def _activation_perms(backend, ref_a, path_a, sa, sb, spec, device, ema, cancel=None):
    import torch

    from app.core.craft.ladder import _own
    from app.core.engine.sampler import SampleParams, sampler

    net, meta = backend.load(ref_a, device=device, ema=ema)
    _own(net)
    bundle = {"model": net, "meta": meta, "backend": backend, "ref": ref_a}
    x0 = []
    for state in (sa, sb):
        backend.load_slot(net, state)
        for seed in range(0, PROBE_SAMPLES, 4):
            p = SampleParams(model_path=path_a, image_size=PROBE_SIZE, steps=PROBE_STEPS,
                             seed=seed, batch_size=4, device=device, ema=ema)
            last = None
            for last in sampler.run(p, bundle=bundle, cancel=cancel):
                pass
            if cancel is not None and cancel():
                raise Cancelled()
            x0.append(last._x.detach())
    x0 = torch.cat(x0)
    betas = torch.as_tensor(backend.schedule(ref_a).trained_betas, dtype=torch.float32)
    xs, ts = noised_inputs(x0, torch.cumprod(1 - betas, 0))
    backend.load_slot(net, sa)
    acts_a = activations(net, spec, xs, ts)
    backend.load_slot(net, sb)
    acts_b = activations(net, spec, xs, ts)
    return match_activations(acts_a, acts_b)


def align_slots(slots: dict, spec: Spec, perms: dict) -> dict:
    """Every slot of a checkpoint reordered by ``perms``; tensors outside the spec as they are."""
    return {name: apply(state, spec, perms) for name, state in slots.items()}
