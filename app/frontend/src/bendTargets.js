// Single source of truth for what a bend's `targets` array actually hits.
//
// The wire format is a flat array mixing group names ("encoder", "attention")
// with exact module ids ("downs.1.0"), matching _resolve_targets in
// app/core/craft/bending.py. This module mirrors that resolution for the UI so
// the map can never claim a layer is targeted when the backend disagrees.

export const GROUPS = ["all", "encoder", "mid", "decoder", "attention", "blocks"];

/** Node ids a single group name expands to, in map order.
 *
 *  Only the main points: skips and the layers inside blocks (`extra`) are
 *  targets you pick one by one, as the backend's groups never include them. */
export function expandGroup(group, nodes) {
  const all = (nodes || []).filter((n) => !n.extra);
  switch (group) {
    case "all": return all.map((n) => n.id);
    case "encoder":
    case "mid":
    case "decoder": return all.filter((n) => n.stage === group).map((n) => n.id);
    case "attention": return all.filter((n) => n.type === "attention").map((n) => n.id);
    case "blocks": return all.filter((n) => n.type === "block").map((n) => n.id);
    default: return [];
  }
}

export function isGroup(t) {
  return GROUPS.includes(t);
}

/**
 * Resolve a targets array to the set of node ids it affects.
 *
 * Unknown strings are dropped rather than added: the backend silently ignores
 * ids it cannot find, so adding them here would ring a node that never bends.
 */
export function resolveTargets(targets, nodes) {
  const known = new Set((nodes || []).map((n) => n.id));
  const ids = new Set();
  for (const t of targets || []) {
    if (isGroup(t)) expandGroup(t, nodes).forEach((id) => ids.add(id));
    else if (known.has(t)) ids.add(t);
  }
  return ids;
}

/**
 * Which bends hold each layer: id -> [{ n, active }], `n` being the bend's
 * 1-based place in the stack. That number is what the stack cards show, so the
 * map can say "bend 1 is here" instead of only "something else is here".
 * `skipId` leaves out the focused bend, which the map draws on its own.
 */
export function holdersOf(bends, nodes, skipId) {
  const out = new Map();
  (bends || []).forEach((b, i) => {
    if (b.id === skipId) return;
    resolveTargets(b.targets, nodes).forEach((id) => {
      const list = out.get(id) || [];
      list.push({ n: i + 1, active: b.active !== false });
      out.set(id, list);
    });
  });
  return out;
}

/** The bends holding any of `ids`, each once, in stack order. */
export function holdersIn(holders, ids) {
  const seen = new Map();
  (ids || []).forEach((id) => (holders?.get(id) || []).forEach((h) => seen.set(h.n, h)));
  return [...seen.values()].sort((a, b) => a.n - b.n);
}

/** True when an active bend other than the focused one holds `id`. */
export function heldByActive(holders, id) {
  return !!holders?.get(id)?.some((h) => h.active);
}

function orderTargets(targets, nodes) {
  const index = new Map((nodes || []).map((n, i) => [n.id, i]));
  const groups = targets.filter(isGroup);
  const ids = targets.filter((t) => !isGroup(t))
    .sort((a, b) => (index.get(a) ?? 0) - (index.get(b) ?? 0));
  return [...groups, ...ids];
}

/**
 * Toggle `ids` on a bend, returning `{ targets, expanded }`.
 *
 * `force` true adds, false removes, undefined toggles off the first id's
 * current state. Removing a layer that is only covered by a group materialises
 * every group on the bend into explicit ids first -- clicking a layer has to
 * visibly turn it off, and the wire format has no way to say "encoder except
 * this one". The group chip re-collapses it.
 */
export function toggleNodeTargets(bend, ids, nodes, { force } = {}) {
  const wanted = (ids || []).filter(Boolean);
  if (!wanted.length) return { targets: bend.targets || [], expanded: [] };

  const current = resolveTargets(bend.targets, nodes);
  const add = force === undefined ? !current.has(wanted[0]) : !!force;
  let targets = [...(bend.targets || [])];
  const expanded = [];

  if (add) {
    wanted.forEach((id) => {
      if (!current.has(id) && !targets.includes(id)) targets.push(id);
    });
    return { targets: orderTargets(targets, nodes), expanded };
  }

  // Removing: any id held by a group forces every group on this bend open.
  const heldByGroup = wanted.some((id) => current.has(id) && !targets.includes(id));
  if (heldByGroup) {
    const flat = [];
    targets.forEach((t) => {
      if (isGroup(t)) { expanded.push(t); expandGroup(t, nodes).forEach((id) => flat.push(id)); }
      else flat.push(t);
    });
    targets = [...new Set(flat)];
  }
  const drop = new Set(wanted);
  targets = targets.filter((t) => !drop.has(t));
  return { targets: orderTargets(targets, nodes), expanded };
}

/** Explicit (non-group) ids on a bend, in map order. */
export function explicitTargets(bend, nodes) {
  const known = new Set((nodes || []).map((n) => n.id));
  return orderTargets((bend.targets || []).filter((t) => !isGroup(t) && known.has(t)), nodes);
}
