const GROUP_LABELS = {
  all: "all layers",
  encoder: "encoder",
  mid: "mid",
  decoder: "decoder",
  attention: "attention",
  blocks: "blocks",
};

function formatTarget(t) {
  return GROUP_LABELS[t] || t;
}

/** Where a bend acts, in a few words: group names, or the first few layers. */
export function formatTargets(targets) {
  // Not "all layers": _resolve_targets in app/core/craft/bending.py expands an
  // empty list to the empty set, so a bend with no targets hooks nothing at
  // all. Reading it the other way round made a bend look like it was doing the
  // most it possibly could when it was doing nothing.
  if (!targets?.length) return "no layers yet";
  if (targets.length <= 3) return targets.map(formatTarget).join(", ");
  return `${targets.slice(0, 3).map(formatTarget).join(", ")} +${targets.length - 3} more`;
}

/** Index the op catalog by name so a bend can describe itself accurately. */
export function opIndex(ops) {
  return Object.fromEntries((ops || []).map((o) => [o.name, o]));
}

/** The headline number for a bend, or null when the op has no parameters.
 *
 *  The catalog names the headline parameter (``amount_param``); guessing from
 *  the params got it wrong for most ops, which use std / factor / angle / mix /
 *  gain / t / k rather than anything called "amount".
 */
export function bendAmount(b, def) {
  const params = b?.params || {};
  const key = def?.amount_param
    || Object.keys(params).find((k) => ["amount", "value", "strength"].includes(k))
    || Object.keys(params)[0];
  const v = key != null ? params[key] : null;
  return v == null ? null : String(v);
}

/** "Clamp -0.8…0.8", "Shift (roll) 6, 0", "Multiply 0.65": the op and the
 *  numbers that matter, for places with room for one short line per bend. */
export function bendHeadline(b, def) {
  const name = def?.label || b?.op || "bend";
  const p = b?.params || {};
  if (p.min != null && p.max != null && def?.params?.some((x) => x.name === "min")) {
    return `${name} ${p.min}…${p.max}`;
  }
  if (p.dx != null || p.dy != null) return `${name} ${p.dx ?? 0}, ${p.dy ?? 0}`;
  const amount = bendAmount(b, def);
  return amount != null ? `${name} ${amount}` : name;
}

function bendLine(b, i, opMap) {
  const def = opMap?.[b.op];
  const amount = bendAmount(b, def);
  const amt = amount != null ? ` ${amount}` : "";
  const name = def?.label || b.op || "bend";
  const sched = (b.step_start != null && b.step_end != null && (b.step_start > 0 || b.step_end < 1))
    ? ` · steps ${Math.round(b.step_start * 100)}–${Math.round(b.step_end * 100)}%`
    : "";
  const off = b.active === false ? "  (off)" : (
    (b.targets || []).length === 0 ? "  (no effect until a layer is chosen)" : "");
  return `${i + 1}. ${name}${amt} → ${formatTargets(b.targets)}${sched}${off}`;
}

/** Multi-line synopsis of a saved bend preset, for tooltips. */
export function bendPresetSynopsis(preset, ops) {
  const bends = preset?.bends || [];
  if (!bends.length) return "Empty bend stack — no layers targeted.";
  const opMap = Array.isArray(ops) ? opIndex(ops) : ops;
  const active = bends.filter((b) => b.active !== false).length;
  const head = active === bends.length
    ? `${bends.length} bend${bends.length === 1 ? "" : "s"}:`
    : `${active} active of ${bends.length} bends:`;
  // A preset's own description first, where it has one: on the starter recipes
  // that sentence is the whole point, and the op list underneath is the detail.
  const notes = (preset?.notes || "").trim();
  const body = `${head}\n${bends.map((b, i) => bendLine(b, i, opMap)).join("\n")}`;
  return notes ? `${notes}\n\n${body}` : body;
}

/** One-line version for inline hints. */
export function bendPresetSummary(preset, ops) {
  const bends = preset?.bends || [];
  if (!bends.length) return "empty stack";
  const opMap = Array.isArray(ops) ? opIndex(ops) : ops;
  const names = bends
    .filter((b) => b.active !== false)
    .map((b) => opMap?.[b.op]?.label || b.op);
  const targets = new Set();
  bends.forEach((b) => (b.targets || []).forEach((t) => targets.add(formatTarget(t))));
  const where = targets.size ? ` on ${[...targets].slice(0, 2).join(", ")}` : "";
  return `${names.join(" → ") || "no active bends"}${where}`;
}
