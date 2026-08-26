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

function formatTargets(targets) {
  if (!targets?.length) return "all layers";
  if (targets.length <= 3) return targets.map(formatTarget).join(", ");
  return `${targets.slice(0, 3).map(formatTarget).join(", ")} +${targets.length - 3} more`;
}

/** Index the op catalog by name so a bend can describe itself accurately. */
export function opIndex(ops) {
  return Object.fromEntries((ops || []).map((o) => [o.name, o]));
}

function bendLine(b, i, opMap) {
  const def = opMap?.[b.op];
  const params = b.params || {};
  // The catalog knows which parameter is the headline one; guessing got it
  // wrong for most ops (they use std / factor / angle / mix / gain / t / k).
  const amountKey = def?.amount_param
    || Object.keys(params).find((k) => ["amount", "value", "strength"].includes(k))
    || Object.keys(params)[0];
  const amount = amountKey != null ? params[amountKey] : null;
  const amt = amount != null ? ` ${amount}` : "";
  const name = def?.label || b.op || "bend";
  const sched = (b.step_start != null && b.step_end != null && (b.step_start > 0 || b.step_end < 1))
    ? ` · steps ${Math.round(b.step_start * 100)}–${Math.round(b.step_end * 100)}%`
    : "";
  const off = b.active === false ? "  (off)" : "";
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
  return `${head}\n${bends.map((b, i) => bendLine(b, i, opMap)).join("\n")}`;
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
