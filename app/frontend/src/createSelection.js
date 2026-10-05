// What Create's Generate panel bends and blends the whole canvas with: a bend
// preset, a merge recipe and the B that recipe blends in. Kept in localStorage
// (the bend preset always was) so Restore can set it from screens Create is not
// mounted beside, and announced so a mounted panel follows.
import { useEffect, useState } from "react";
import { api } from "./api.js";

const KEYS = { bend: "kiln.genBendPreset", merge: "kiln.genMergeRecipe", mergeB: "kiln.genMergeB" };
const EVENT = "kiln:create-selection";

function read(key) {
  try { return localStorage.getItem(key) || ""; } catch { return ""; }
}

export function loadSelection() {
  return { bend: read(KEYS.bend), merge: read(KEYS.merge), mergeB: read(KEYS.mergeB) };
}

export function storeSelection(patch) {
  Object.entries(patch).forEach(([k, v]) => {
    if (!KEYS[k]) return;
    try { localStorage.setItem(KEYS[k], v || ""); } catch { /* kept for this session only */ }
  });
  window.dispatchEvent(new CustomEvent(EVENT));
}

/** Create's selection, following changes made anywhere (Restore included). */
export function useCreateSelection() {
  const [sel, setSel] = useState(loadSelection);
  useEffect(() => {
    const follow = () => setSel(loadSelection());
    window.addEventListener(EVENT, follow);
    return () => window.removeEventListener(EVENT, follow);
  }, []);
  return [sel, storeSelection];
}

/** The model a card was sampled on. A ladder rung names none: its A is in `merge`. */
export const cardModelPath = (card) => card?.model_path || card?.merge?.model_a || "";

/** Put a card's bend preset and merge recipe back into Create's selection.
 *
 *  Restoring an image's settings has to restore what changed its pixels: a
 *  card without a bend preset clears the one selected, or the next run would
 *  be bent when the image was not. Returns what could not come back, for the
 *  toast: a preset or recipe deleted since, or bends and blends that were
 *  never saved as one.
 */
export async function restoreSelection(card) {
  if (!card) return [];
  const [bends, recipes] = await Promise.all([
    api.get("/library/bends").catch(() => []),
    api.get("/library/recipes").catch(() => []),
  ]);
  const missing = [];
  let bend = "";
  if (card.bend_preset) {
    if (bends.some((b) => b.name === card.bend_preset)) bend = card.bend_preset;
    else missing.push(`bend preset “${card.bend_preset}” is no longer saved`);
  } else if (card.bends?.length) {
    missing.push("its bends were never saved as a preset");
  }
  let merge = "";
  let mergeB = "";
  if (card.merge) {
    if (card.merge_recipe && recipes.some((r) => r.name === card.merge_recipe)) {
      merge = card.merge_recipe;
      mergeB = card.merge.model_b || "";
    } else {
      missing.push(card.merge_recipe
        ? `merge recipe “${card.merge_recipe}” is no longer saved`
        : "its blend was never saved as a merge recipe");
    }
  }
  storeSelection({ bend, merge, mergeB });
  return missing;
}

/** The toast for a restore: what came back, then what did not. */
export function restoredMessage(label, missing) {
  const head = `Settings restored — ${label || "sampler updated"}`;
  return missing.length ? `${head}. Not restored: ${missing.join("; ")}` : head;
}
