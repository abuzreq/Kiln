import { newBendId } from "./bendStack.js";

// The dice. Rolls the same kind of bend the Discoveries explorer tries
// (random_bend in app/core/craft/explore.py): any op, its numbers drawn across
// their whole range, a run window jittered around the op's usual one. A dice
// roll and an explored bend are then the same sort of thing, and a roll you
// like can be found again by exploring.
//
// Where it acts differs on purpose. The explorer only takes named groups; here,
// half the rolls take a short run of neighbouring layers instead, since the map
// is right there to show (and change) what was picked.

const GROUP_WEIGHTS = [
  ["encoder", 0.25], ["mid", 0.25], ["decoder", 0.25],
  ["attention", 0.1], ["blocks", 0.05], ["all", 0.05],
];
const RUN_MAX = 4;
const MIN_WINDOW = 0.15;

const clamp01 = (x) => Math.max(0, Math.min(1, x));
const uniform = (lo, hi) => lo + Math.random() * (hi - lo);
const pick = (list) => list[Math.floor(Math.random() * list.length)];
const round = (x, n = 4) => Number(x.toFixed(n));

function weighted(pairs) {
  let r = Math.random() * pairs.reduce((a, [, w]) => a + w, 0);
  for (const [v, w] of pairs) {
    r -= w;
    if (r <= 0) return v;
  }
  return pairs[pairs.length - 1][0];
}

function randParam(p) {
  if (p.kind === "select") return pick(p.options?.length ? p.options : [p.default]);
  if (p.min == null || p.max == null) return p.default;
  if (p.kind === "int") {
    const step = Math.max(1, Math.round(p.step || 1));
    const n = Math.floor((p.max - p.min) / step);
    return p.min + step * Math.floor(Math.random() * (n + 1));
  }
  return round(uniform(p.min, p.max));
}

/** Every number of `op`, rolled. A min/max pair comes back in order. */
export function randomParams(op) {
  const params = {};
  (op?.params || []).forEach((p) => { params[p.name] = randParam(p); });
  if (typeof params.min === "number" && typeof params.max === "number" && params.min > params.max) {
    [params.min, params.max] = [params.max, params.min];
  }
  return params;
}

/** The op's usual window, nudged, and never narrower than MIN_WINDOW. */
export function randomWindow(op) {
  const s = op?.schedule || {};
  let start = clamp01((s.start ?? 0) + uniform(-0.15, 0.15));
  let end = clamp01((s.end ?? 1) + uniform(-0.15, 0.15));
  if (end - start < MIN_WINDOW) {
    const mid = (start + end) / 2;
    start = clamp01(mid - MIN_WINDOW / 2);
    end = clamp01(mid + MIN_WINDOW / 2);
    if (end - start < MIN_WINDOW) [start, end] = start < 0.5 ? [0, MIN_WINDOW] : [1 - MIN_WINDOW, 1];
  }
  return { step_start: round(start, 3), step_end: round(end, 3) };
}

/** A named group, or a run of one to RUN_MAX neighbouring layers. */
export function randomTargets(nodes) {
  if (!nodes?.length || Math.random() < 0.5) return [weighted(GROUP_WEIGHTS)];
  const len = 1 + Math.floor(Math.random() * Math.min(RUN_MAX, nodes.length));
  const at = Math.floor(Math.random() * (nodes.length - len + 1));
  return nodes.slice(at, at + len).map((n) => n.id);
}

/** A whole new bend: op, numbers, window and where it acts. */
export function randomBend(ops, nodes) {
  const op = pick(ops || []);
  if (!op) return null;
  return {
    id: newBendId(),
    op: op.name,
    params: randomParams(op),
    targets: randomTargets(nodes),
    ...randomWindow(op),
    active: true,
  };
}

/**
 * `bend` re-rolled in place: a new op, numbers and window. Where it acts is
 * kept -- picking layers is the deliberate part -- unless it acts nowhere yet,
 * in which case that is rolled too.
 */
export function rerollBend(bend, ops, nodes) {
  const fresh = randomBend(ops, nodes);
  if (!fresh) return bend;
  const targets = bend.targets?.length ? bend.targets : fresh.targets;
  return { ...fresh, id: bend.id, active: bend.active, targets };
}
