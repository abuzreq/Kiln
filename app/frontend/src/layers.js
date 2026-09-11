/** Canvas entities: raster layers and inpaint masks.
 *
 *  Modelled on InvokeAI's controlLayers store
 *  (`invokeai/frontend/web/src/features/controlLayers/store/types.ts`), which
 *  splits canvas entities into typed groups over a common base of
 *  `{ id, name, isEnabled, isLocked }`. Two of their groups apply to Kiln:
 *
 *    raster_layer   image content, composited bottom-to-top to form the canvas
 *    inpaint_mask   where a fill is allowed to change things
 *
 *  Two deliberate departures:
 *
 *  - InvokeAI raster layers hold `objects[]` — vector-ish records rendered
 *    through konva. Kiln's layer content arrives as a finished raster from the
 *    sampler, so a raster layer holds an image and only masks keep stroke
 *    lists (see selection.js, which does the replaying).
 *  - No `position`. InvokeAI can move, scale and transform an entity; Kiln has
 *    no tooling for that, and every layer is canvas-aligned. Adding a transform
 *    would be a bigger job than the whole panel.
 */

import { rasterize } from "./selection.js";

let counter = 0;
const nextId = () => `e${(counter += 1)}_${Math.random().toString(36).slice(2, 8)}`;

function entityBase(name) {
  return { id: nextId(), name, enabled: true, locked: false };
}

/** Content. `image` is a full-canvas RGBA data URL, transparent where the
 *  layer contributes nothing — which is how a region fill stacks over what is
 *  already there instead of hiding it. */
export function newRasterLayer({ name, image = null, card = null }) {
  return { ...entityBase(name || "Layer"), type: "raster", opacity: 1, image, card };
}

/** Where a fill may change things. `params` is Kiln's equivalent of InvokeAI's
 *  per-mask `noiseLevel` / `denoiseLimit`: settings that belong to the region
 *  rather than to a global slider. */
export function newInpaintMask({ name } = {}) {
  return {
    ...entityBase(name || "Inpaint Mask"),
    type: "mask",
    strokes: [],
    params: { change: 0.65, feather: 8, harmonize: 2, bendPreset: "" },
  };
}

/** A fresh document. Ships with one empty inpaint mask already present, as
 *  InvokeAI does — it makes the concept discoverable instead of something you
 *  have to know to create. */
export function newDocument(ground) {
  return {
    rasterLayers: ground
      ? [newRasterLayer({ name: "Background", image: ground })]
      : [],
    inpaintMasks: [newInpaintMask({ name: "Inpaint Mask 1" })],
  };
}

/** Unique "Name n" within a group. */
export function uniqueName(list, stem) {
  const taken = new Set(list.map((e) => e.name));
  for (let i = 1; i < 999; i += 1) {
    const n = `${stem} ${i}`;
    if (!taken.has(n)) return n;
  }
  return stem;
}

/** Keep a proposed name, numbering it only if it is already taken.
 *
 *  Fill layers are named after the model that made them, so a second attempt
 *  with the same model would otherwise produce two rows reading exactly alike
 *  — and telling attempts apart is the entire point of stacking them.
 */
export function uniqueFrom(list, name) {
  const taken = new Set(list.map((e) => e.name));
  if (!taken.has(name)) return name;
  // From 2, because the unsuffixed row is the first one.
  for (let i = 2; i < 999; i += 1) {
    const n = `${name} ${i}`;
    if (!taken.has(n)) return n;
  }
  return name;
}

export const visibleRasters = (layers) => layers.filter((l) => l.enabled && l.image);
export const activeMasks = (masks) => masks.filter((m) => m.enabled && m.strokes.length);

function loadImage(src) {
  return new Promise((resolve, reject) => {
    const im = new Image();
    im.onload = () => resolve(im);
    im.onerror = () => reject(new Error("Could not load a layer image"));
    im.src = src;
  });
}

/** Flatten the visible raster layers, bottom to top, into one image.
 *
 *  This is the canvas. Everything downstream — export, upscale, post-process,
 *  the init image for the next fill, what the wand samples — reads the result
 *  rather than any single layer.
 */
export async function flattenLayers(layers, size) {
  const vis = visibleRasters(layers);
  if (!vis.length) return null;
  const imgs = await Promise.all(vis.map((l) => loadImage(l.image)));
  const w = size?.w || Math.max(...imgs.map((i) => i.naturalWidth || i.width));
  const h = size?.h || Math.max(...imgs.map((i) => i.naturalHeight || i.height));
  if (!w || !h) return null;
  const c = document.createElement("canvas");
  c.width = w;
  c.height = h;
  const ctx = c.getContext("2d");
  imgs.forEach((im, i) => {
    ctx.globalAlpha = vis[i].opacity ?? 1;
    ctx.drawImage(im, 0, 0, w, h);
  });
  ctx.globalAlpha = 1;
  return c.toDataURL("image/png");
}

/** Rasterise every enabled mask onto one canvas — their union is the mask the
 *  backend is sent. Overlapping masks take the stronger alpha. */
export function compositeMasks(masks, target, w, h) {
  if (!target || !w || !h) return;
  if (target.width !== w || target.height !== h) {
    target.width = w;
    target.height = h;
  }
  const ctx = target.getContext("2d");
  ctx.globalAlpha = 1;
  ctx.globalCompositeOperation = "source-over";
  ctx.clearRect(0, 0, w, h);
  const scratch = document.createElement("canvas");
  for (const m of masks) {
    if (!m.enabled || !m.strokes.length) continue;
    rasterize(m, scratch, w, h);
    ctx.drawImage(scratch, 0, 0);
  }
}

/** Punch a fill result down to just the region the mask allowed.
 *
 *  A fill comes back as a whole frame with the region changed. Stored as-is it
 *  would be an opaque layer hiding everything beneath, which makes the stack a
 *  linear history rather than a composition. So the frame goes transparent
 *  outside the mask.
 *
 *  The alpha is binary, not the mask's own gradient, and that matters. The
 *  backend has *already* blended the fill against the original at mask
 *  strength, so the returned frame is the finished result. Carrying the
 *  gradient into the layer alpha would blend it a second time on composite: a
 *  region masked at half strength would land at a quarter. Binary alpha makes
 *  the composite reproduce the returned frame exactly.
 *
 *  A hard alpha edge draws no seam, because the softness lives in the pixel
 *  values rather than in the alpha — out at the edge the returned frame has
 *  already decayed to the original, so cutting there changes nothing visible.
 *  It also makes layer opacity meaningful: it now fades the whole fill back
 *  towards what was underneath.
 */
export async function punchToMask(frameSrc, maskCanvas) {
  if (!frameSrc || !maskCanvas?.width) return frameSrc;
  const im = await loadImage(frameSrc);
  const w = im.naturalWidth || im.width;
  const h = im.naturalHeight || im.height;
  const c = document.createElement("canvas");
  c.width = w;
  c.height = h;
  const ctx = c.getContext("2d");
  ctx.drawImage(im, 0, 0, w, h);
  const frame = ctx.getImageData(0, 0, w, h);

  // The mask may have been built at a different size; scale it to the frame.
  const ms = document.createElement("canvas");
  ms.width = w;
  ms.height = h;
  ms.getContext("2d").drawImage(maskCanvas, 0, 0, w, h);
  const mask = ms.getContext("2d").getImageData(0, 0, w, h);

  for (let i = 0; i < w * h; i += 1) {
    frame.data[i * 4 + 3] = mask.data[i * 4 + 3] > 0 ? 255 : 0;
  }
  ctx.putImageData(frame, 0, 0);
  return c.toDataURL("image/png");
}

/** A raster layer's own alpha, as a mask stroke cache.
 *
 *  This is what makes "use as mask" work: a fill layer's alpha is exactly the
 *  region that fill covered, so promoting it selects that area again and the
 *  next fill can rework the same place with different settings.
 */
export async function layerAlphaToCache(image) {
  const im = await loadImage(image);
  const w = im.naturalWidth || im.width;
  const h = im.naturalHeight || im.height;
  const c = document.createElement("canvas");
  c.width = w;
  c.height = h;
  const ctx = c.getContext("2d");
  ctx.drawImage(im, 0, 0);
  const d = ctx.getImageData(0, 0, w, h);
  for (let i = 0; i < w * h; i += 1) {
    const o = i * 4;
    d.data[o] = 255;
    d.data[o + 1] = 122;
    d.data[o + 2] = 69;
  }
  ctx.putImageData(d, 0, 0);
  return { cache: c, w, h };
}

/** Move an entity within its group. Returns a new array. */
export function reorder(list, id, delta) {
  const i = list.findIndex((e) => e.id === id);
  const j = i + delta;
  if (i < 0 || j < 0 || j >= list.length) return list;
  const out = [...list];
  [out[i], out[j]] = [out[j], out[i]];
  return out;
}
