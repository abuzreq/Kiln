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
 *  - No `position`. InvokeAI can move, scale and transform an entity; Kiln's
 *    layers are all canvas-aligned. A mask can be moved, but that is a stroke
 *    in its list (see selection.js `moveStroke`), not a field on the entity.
 */

import { rasterize, MASK_RGB } from "./selection.js";

let counter = 0;
const nextId = () => `e${(counter += 1)}_${Math.random().toString(36).slice(2, 8)}`;

// No `locked` here, unlike InvokeAI. The first cut carried one, and it stopped
// the brush and nothing else -- delete, move, hide, clear and invert all went
// through -- which made it a control that mostly did nothing in a rail with no
// room for one.
function entityBase(name) {
  return { id: nextId(), name, enabled: true };
}

/** Content. `image` is a full-canvas RGBA data URL, transparent where the
 *  layer contributes nothing — which is how a region fill sits over what is
 *  on the layers below instead of hiding it.
 *
 *  Layers are the user's. Kiln never adds one on its own: a run writes into
 *  the active layer, and how the work is split across layers is how the user
 *  chose to structure their exploration. `card` is the recipe of whatever last
 *  wrote into the layer.
 */
export function newRasterLayer({ name, image = null, card = null, enabled = true }) {
  return { ...entityBase(name || "Layer"), type: "raster", opacity: 1, image, card, enabled };
}

/** Where a fill may change things. `params` is Kiln's equivalent of InvokeAI's
 *  per-mask `noiseLevel` / `denoiseLimit`: settings that belong to the region
 *  rather than to a global slider. */
export function newInpaintMask({ name } = {}) {
  return {
    ...entityBase(name || "Inpaint Mask"),
    type: "mask",
    // Two switches, unlike a layer's one. `enabled` is whether the next run
    // takes the mask into account; `visible` is whether its hatching is drawn.
    // They are independent: a mask can be off but shown (kept in view while
    // another area is worked on) or on but hidden (out of the way of judging
    // the pixels underneath).
    visible: true,
    strokes: [],
    params: { change: 0.65, feather: 8, harmonize: 2, bendPreset: "" },
  };
}

/** Whether a mask's hatching is drawn. Masks made before the flag existed
 *  have no `visible` and count as shown. */
export const maskShown = (m) => m?.visible !== false;

/** A fresh document: one layer to work in, one empty inpaint mask.
 *
 *  Both exist from the start so that generating and painting need nothing set
 *  up first -- there is always an active layer for a run to land in, and the
 *  mask makes the concept discoverable instead of something you have to know
 *  to create (InvokeAI does the same with its mask). */
export function newDocument(ground) {
  return {
    rasterLayers: [newRasterLayer({ name: "Layer 1", image: ground })],
    inpaintMasks: [newInpaintMask({ name: "Inpaint Mask 1" })],
  };
}

/** A transparent-black PNG of the given size — an empty layer. */
export function blankImage(w, h) {
  const c = document.createElement("canvas");
  c.width = w;
  c.height = h;
  // New canvases are already (0,0,0,0); clear so that stays true if a browser
  // ever changes the default.
  c.getContext("2d").clearRect(0, 0, w, h);
  return c.toDataURL("image/png");
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

/** Rasterise the given masks onto one canvas. Overlapping masks take the
 *  stronger alpha. Callers choose the set: the masks that are on make the
 *  mask the backend is sent; the masks that are shown make the display. */
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
    if (!m.strokes.length) continue;
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
 *
 *  `maskSrc` is the very mask the backend was sent (see maskDataUrlFrom): white
 *  where the fill happened, opaque everywhere. Punching with that rather than
 *  with the live overlay matters, because the overlay stays paintable while a
 *  fill runs. The first cut read the overlay when the job *finished*, so
 *  hiding the mask mid-run punched the result to nothing and the fill was
 *  silently lost.
 */
export async function punchToMask(frameSrc, maskSrc) {
  if (!frameSrc || !maskSrc) return frameSrc;
  const [im, mk] = await Promise.all([loadImage(frameSrc), loadImage(maskSrc)]);
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
  ms.getContext("2d").drawImage(mk, 0, 0, w, h);
  const mask = ms.getContext("2d").getImageData(0, 0, w, h);

  for (let i = 0; i < w * h; i += 1) {
    frame.data[i * 4 + 3] = mask.data[i * 4] > 0 ? 255 : 0;
  }
  ctx.putImageData(frame, 0, 0);
  return c.toDataURL("image/png");
}

/** `top` drawn over `base`, both stretched to the canvas. How a fill lands in
 *  a layer that already has content: the punched region replaces what the
 *  layer had there and leaves the rest of the layer alone. */
export async function compositeOnto(baseSrc, topSrc, w, h) {
  const [base, top] = await Promise.all([
    baseSrc ? loadImage(baseSrc) : null,
    loadImage(topSrc),
  ]);
  const c = document.createElement("canvas");
  c.width = w;
  c.height = h;
  const ctx = c.getContext("2d");
  if (base) ctx.drawImage(base, 0, 0, w, h);
  ctx.drawImage(top, 0, 0, w, h);
  return c.toDataURL("image/png");
}

/** An image placed on a canvas of a different size: scaled to fit, centred,
 *  transparent around it. Layers are canvas-aligned and flatten stretches
 *  every image to the canvas, so an asset of another aspect has to be fitted
 *  before it becomes a layer's image or it would be distorted. */
export async function fitIntoCanvas(src, w, h) {
  const im = await loadImage(src);
  const iw = im.naturalWidth || im.width;
  const ih = im.naturalHeight || im.height;
  const c = document.createElement("canvas");
  c.width = w;
  c.height = h;
  const k = Math.min(w / iw, h / ih);
  const dw = Math.round(iw * k);
  const dh = Math.round(ih * k);
  c.getContext("2d").drawImage(im, Math.round((w - dw) / 2), Math.round((h - dh) / 2), dw, dh);
  return c.toDataURL("image/png");
}

/** Natural size of an image source. */
export async function imageSize(src) {
  const im = await loadImage(src);
  return { w: im.naturalWidth || im.width, h: im.naturalHeight || im.height };
}

/** The mask as the backend wants it: white where the fill should happen.
 *
 *  Reads an overlay canvas whose alpha carries mask strength (the union of
 *  every enabled mask), and returns null when nothing on it is painted.
 */
export function maskDataUrlFrom(c) {
  if (!c?.width || !c?.height) return null;
  const ctx = c.getContext("2d");
  const img = ctx.getImageData(0, 0, c.width, c.height);
  const { data } = img;
  let painted = false;
  for (let i = 0; i < data.length; i += 4) {
    const a = data[i + 3];
    if (a > 8) painted = true;
    data[i] = data[i + 1] = data[i + 2] = a;
    data[i + 3] = 255;
  }
  if (!painted) return null;
  const out = document.createElement("canvas");
  out.width = c.width;
  out.height = c.height;
  out.getContext("2d").putImageData(img, 0, 0);
  return out.toDataURL("image/png");
}

/** A raster layer's own alpha, as a mask stroke cache.
 *
 *  This is what makes "Select area" work: a layer that has only ever taken
 *  fills is transparent everywhere else, so its alpha is exactly the ground
 *  those fills covered, and promoting it selects that area again.
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
  const [r, g, b] = MASK_RGB;
  for (let i = 0; i < w * h; i += 1) {
    const o = i * 4;
    d.data[o] = r;
    d.data[o + 1] = g;
    d.data[o + 2] = b;
  }
  ctx.putImageData(d, 0, 0);
  return { cache: c, w, h };
}

/** A grayscale mask picture (white = selected), as a mask stroke cache.
 *
 *  What the shape generator returns. The red channel becomes the alpha, so a
 *  softened edge is partial strength exactly as a soft brush is.
 */
export async function maskUrlToCache(url) {
  const im = await loadImage(url);
  const w = im.naturalWidth || im.width;
  const h = im.naturalHeight || im.height;
  const c = document.createElement("canvas");
  c.width = w;
  c.height = h;
  const ctx = c.getContext("2d");
  ctx.drawImage(im, 0, 0);
  const d = ctx.getImageData(0, 0, w, h);
  const [r, g, b] = MASK_RGB;
  for (let i = 0; i < w * h; i += 1) {
    const o = i * 4;
    const a = d.data[o];
    d.data[o] = r;
    d.data[o + 1] = g;
    d.data[o + 2] = b;
    d.data[o + 3] = a > 8 ? a : 0;
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
