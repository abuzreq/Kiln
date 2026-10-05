// Merge recipes: a saved blend of two models, the way a bend preset is a saved
// stack. One keeps the mix (method, alpha or block weights), the model it was
// made on (model_a, a hint) and the partner (model_b). The Merge tab saves and
// loads them; Create blends a recipe's B into the selected model with "Merge
// with". Recipes saved before they kept B have no model_b.

export const STAGE_IDS = ["encoder", "mid", "decoder"];
// The methods that merge A's and B's changes from a base model; they carry a
// density and a strength as well as alpha, and need the base to mean anything.
export const BASE_METHOD_IDS = ["task_arithmetic", "ties", "dare_ties"];
export const isBaseMethod = (m) => BASE_METHOD_IDS.includes(m);
const METHOD_LABELS = {
  linear: "Linear", slerp: "Slerp", blockwise: "Block-wise",
  task_arithmetic: "Task arithmetic", ties: "TIES", dare_ties: "DARE-TIES",
};
export const methodLabel = (m) => METHOD_LABELS[m] || m;

export const round4 = (v) => Math.round(v * 1e4) / 1e4;
export const pct = (v) => {
  const p = round4(v * 100);
  return `${Number.isInteger(p) ? p : p.toFixed(1)}%`;
};

/** A blend written the way the server's ladder.recipe writes it, so keys agree. */
export function recipeOf(method, alpha, blocks, density = 1, strength = 1) {
  if (method === "blockwise") {
    return {
      method,
      block_weights: Object.fromEntries(STAGE_IDS.map((s) => [s, round4(blocks?.[s] ?? 0.5)])),
    };
  }
  if (isBaseMethod(method)) {
    return {
      method, alpha: round4(alpha ?? 0.5), density: round4(density ?? 1), strength: round4(strength ?? 1),
    };
  }
  return { method, alpha: round4(alpha ?? 0.5) };
}

/** The mix a saved recipe holds, in recipeOf's shape. */
export const mixOf = (r) => recipeOf(r.method || "linear", r.alpha, r.block_weights, r.density, r.strength);

/** "Slerp · 70% A / 30% B", the three stages of a block-wise mix, or a merge
 *  from a base with its density and strength. ``onto`` is "A" or "B" when the
 *  base is that model: TIES from it. */
export function describeMix(r, onto = null) {
  if (r.method === "blockwise") {
    const w = r.block_weights;
    return `Block-wise · B ${pct(w.encoder)} encoder, ${pct(w.mid)} mid, ${pct(w.decoder)} decoder`;
  }
  if (onto && isBaseMethod(r.method)) {
    return `${methodLabel(r.method)} onto ${onto} · ${pct(r.density)} of weights moved · strength ${r.strength}`;
  }
  const head = `${methodLabel(r.method)} · ${pct(1 - r.alpha)} A / ${pct(r.alpha)} B`;
  if (!isBaseMethod(r.method)) return head;
  return `${head} · ${r.method === "task_arithmetic" ? "" : `density ${pct(r.density)} · `}strength ${r.strength}`;
}

/** A model's name from its path, for models the list may not hold. */
export function modelName(path, models) {
  if (!path) return "";
  const m = (models || []).find((x) => x.path === path);
  return m?.name || String(path).split(/[\\/]/).pop().replace(/\.(pt|ckpt|safetensors)$/, "");
}

/** One line for a recipe: its mix and its partner. */
export function recipeSummary(r, models) {
  const mix = describeMix(mixOf(r));
  return r.model_b ? `${mix} · with ${modelName(r.model_b, models)}` : `${mix} · partner not recorded`;
}

// Library ▸ Merges opens a recipe in the Merge tab, which may not be mounted
// yet: the request waits here until the tab takes it.
const OPEN_KEY = "kiln.mergeOpen";
const OPEN_EVENT = "kiln:merge-open";

export function requestMergeOpen(name) {
  try { sessionStorage.setItem(OPEN_KEY, name); } catch { /* the event still carries it */ }
  window.dispatchEvent(new CustomEvent(OPEN_EVENT, { detail: name }));
}

/** The recipe waiting to be opened in Merge, taken off the queue. */
export function takeMergeOpen() {
  try {
    const name = sessionStorage.getItem(OPEN_KEY);
    sessionStorage.removeItem(OPEN_KEY);
    return name || null;
  } catch { return null; }
}

export const MERGE_OPEN_EVENT = OPEN_EVENT;
