// The attention layout of a configurable-attention model, as the Train screen
// edits it. The wire format is the vendor's spec string ("-1:linear,mid:full"):
// `mid` is the bottleneck, `-k` the k-th encoder level above it, and the kind is
// full, linear or window. That string is the single source of truth in the form;
// the per-level pickers are a view of it, so these helpers convert both ways.
//
// Mirrors app/core/backends/xurdif/attn.py, including the canonical order
// (deepest level first, mid last) so a spec built here re-selects the named
// layout the server hands out.

export const KINDS = ["none", "full", "linear", "window"];
export const NONE_SPEC = "none";

/** Spec string -> { "-1": "linear", "mid": "full" }. "none" -> {}. Empty -> null. */
export function parseSpec(spec) {
  if (spec == null) return null;
  const value = String(spec).trim();
  if (!value) return null;
  if (value.toLowerCase() === NONE_SPEC) return {};
  const out = {};
  for (const item of value.split(",")) {
    const [loc, kind] = item.split(":").map((s) => (s || "").trim().toLowerCase());
    if (!loc || !kind) continue;
    out[loc === "mid" ? "mid" : String(parseInt(loc, 10))] = kind;
  }
  return out;
}

/** { "-1": "linear", "mid": "full" } -> "-1:linear,mid:full". Empty -> "none". */
export function buildSpec(map) {
  const entries = Object.entries(map || {}).filter(([, k]) => k && k !== "none");
  const levels = entries
    .filter(([loc]) => loc !== "mid")
    .sort((a, b) => parseInt(a[0], 10) - parseInt(b[0], 10));
  const parts = levels.map(([loc, k]) => `${loc}:${k}`);
  const mid = entries.find(([loc]) => loc === "mid");
  if (mid) parts.push(`mid:${mid[1]}`);
  return parts.length ? parts.join(",") : NONE_SPEC;
}

/** Same layout, spelled canonically. */
export function canonical(spec) {
  const map = parseSpec(spec);
  return map == null ? null : buildSpec(map);
}

/** How many encoder levels a "1,2,2,2" multipliers string describes. */
export function depthOf(mults) {
  if (Array.isArray(mults)) return mults.length;
  return String(mults || "").split(",").map((s) => s.trim()).filter(Boolean).length;
}

/** The encoder locations a network of this depth has, deepest first: ["-1", "-2", ...]. */
export function locationsFor(depth) {
  return Array.from({ length: Math.max(depth, 0) }, (_, i) => String(-(i + 1)));
}

/** Drop any level the network no longer has, after the multipliers shrink. */
export function trimToDepth(spec, depth) {
  const map = parseSpec(spec);
  if (map == null) return spec;
  const kept = {};
  for (const [loc, k] of Object.entries(map)) {
    if (loc === "mid" || -parseInt(loc, 10) <= depth) kept[loc] = k;
  }
  return buildSpec(kept);
}

/** The id of the named layout a spec matches, or "custom". */
export function layoutIdFor(spec, layouts) {
  const c = canonical(spec);
  const hit = (layouts || []).find((l) => l.spec === c);
  return hit ? hit.id : "custom";
}

/** The picker value at one location: "none" when the layout has nothing there. */
export function kindAt(spec, loc) {
  const map = parseSpec(spec) || {};
  return map[loc] || "none";
}

/** A new spec with one location changed. */
export function withKind(spec, loc, kind) {
  const map = { ...(parseSpec(spec) || {}) };
  if (!kind || kind === "none") delete map[loc];
  else map[loc] = kind;
  return buildSpec(map);
}
