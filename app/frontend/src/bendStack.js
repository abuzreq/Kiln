// What may enter the Bend tab's stack. Every consumer (the map, BendEditor,
// bendTargets) assumes Kiln's shape, so a stack is checked once on the way in
// rather than at each of them:
//
//   { id, op, params, targets, step_start, step_end, active }
//
// The one foreign shape worth reading is a Bends JSON bend,
// { path, module_type, module_args, frac?, active? }: the bends-json build
// keeps its stack in that shape under the same localStorage key, and its
// discoveries and presets carry it too. It is converted back with the inverse
// of that branch's LEGACY_OPS table (app/core/craft/bendsjson.py there), which
// also covers what interchange.py here writes. An entry whose operation has no
// Kiln equivalent, or that is not a bend at all, is dropped and counted, so the
// caller can say so instead of losing it silently.
import { isGroup } from "./bendTargets.js";

export const newBendId = () => Math.random().toString(36).slice(2);

// module_type -> [Kiln op, { its argument: Kiln parameter }]. Arguments not
// listed keep their name. The Kiln names themselves are included because
// interchange.py exports them verbatim (only rotate's angle is renamed).
const FROM_MODULE = {
  add_scalar: ["add", { scalar: "value" }],
  add: ["add", {}],
  multiply: ["multiply", { scalar: "value" }],
  invert: ["invert", {}],
  abs: ["abs", {}],
  clamp: ["clamp", {}],
  threshold: ["threshold", { threshold: "t" }],
  normalize: ["normalize", {}],
  noise: ["noise", {}],
  channel_shuffle: ["channel_shuffle", {}],
  grid_rotate: ["rotate", { angle_degrees: "angle" }],
  rotate: ["rotate", { angle_degrees: "angle" }],
  grid_scale: ["scale", { scale_factor: "factor" }],
  scale: ["scale", {}],
  roll: ["roll", {}],
  flip: ["flip", { direction: "axis" }],
  dilation: ["dilate", { kernel_size: "k" }],
  dilate: ["dilate", {}],
  erosion: ["erode", { kernel_size: "k" }],
  erode: ["erode", {}],
  edges: ["gradient", {}],
  gradient: ["gradient", {}],
  fourier_highpass: ["fourier_amplify", {}],
  fourier_amplify: ["fourier_amplify", {}],
};

// Bends JSON spells flip's direction out; Kiln's select takes h / v.
const FLIP_AXIS = { horizontal: "h", vertical: "v" };

const isObject = (v) => v !== null && typeof v === "object" && !Array.isArray(v);
const finite = (v, fallback) => (typeof v === "number" && Number.isFinite(v) ? v : fallback);
const strings = (list) => list.filter((s) => typeof s === "string" && s);

function fromModule(b) {
  if (!Object.hasOwn(FROM_MODULE, b.module_type)) return null;
  const mapping = FROM_MODULE[b.module_type];
  const [op, rename] = mapping;
  const params = {};
  Object.entries(isObject(b.module_args) ? b.module_args : {}).forEach(([k, v]) => { params[rename[k] || k] = v; });
  if (op === "flip" && FLIP_AXIS[params.axis]) params.axis = FLIP_AXIS[params.axis];
  // `path` is one pattern or a list; "@encoder" is how Bends JSON spells a group.
  const paths = typeof b.path === "string" ? [b.path] : Array.isArray(b.path) ? b.path : [];
  const targets = strings(paths).map((p) => (p.startsWith("@") && isGroup(p.slice(1)) ? p.slice(1) : p));
  const frac = Array.isArray(b.frac) && b.frac.length === 2 ? b.frac : [0, 1];
  return {
    id: b.id,
    op,
    params,
    targets,
    step_start: finite(frac[0], 0),
    step_end: finite(frac[1], 1),
    active: b.active !== false,
  };
}

/** One bend in Kiln's shape, or null when it cannot be read as a bend. */
export function normalizeBend(b) {
  if (!isObject(b)) return null;
  let bend = null;
  if (typeof b.op === "string" && b.op) bend = b;
  else if (typeof b.module_type === "string") bend = fromModule(b);
  if (!bend) return null;
  return {
    ...bend,
    id: typeof bend.id === "string" && bend.id ? bend.id : newBendId(),
    params: isObject(bend.params) ? bend.params : {},
    targets: Array.isArray(bend.targets) ? strings(bend.targets) : [],
    step_start: finite(bend.step_start, 0),
    step_end: finite(bend.step_end, 1),
    active: bend.active !== false,
  };
}

/**
 * A stack checked entry by entry: `{ bends, dropped, converted }`, where
 * `dropped` counts entries that could not be read as a Kiln bend and
 * `converted` those read from Bends JSON. A non-array is treated as one bad entry.
 */
export function normalizeStack(list) {
  if (list == null) return { bends: [], dropped: 0, converted: 0 };
  if (!Array.isArray(list)) return { bends: [], dropped: 1, converted: 0 };
  const bends = [];
  let converted = 0;
  list.forEach((b) => {
    const n = normalizeBend(b);
    if (!n) return;
    if (!(typeof b.op === "string" && b.op)) converted += 1;
    bends.push(n);
  });
  return { bends, dropped: list.length - bends.length, converted };
}

/** "1 bend" / "3 bends", for the toasts that report a cleanup. */
export const bendCount = (n, one = "bend", many = `${one}s`) => `${n} ${n === 1 ? one : many}`;
