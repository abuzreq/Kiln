/** Strokes, and how to replay them into a mask.
 *
 *  A mask used to be whatever pixels happened to be sitting on the overlay at
 *  the moment Generate was pressed. That made it impossible to undo precisely,
 *  to invert, to hide without losing, or to survive a resize — and it was
 *  silently thrown away after every fill.
 *
 *  Here it is a list of strokes belonging to an inpaint mask entity (see
 *  layers.js). The overlay is a cache of replaying them, so undo is "drop the
 *  last stroke and replay", and a resize is "replay at the new size".
 *  Coordinates are normalised 0..1 so neither depends on the pixel dimensions
 *  the strokes happened to be drawn at.
 */

/** The mask colour.
 *
 *  Only the alpha of a rasterised mask carries meaning — it is the per-pixel
 *  denoise strength, and getMaskDataUrl overwrites the colour channels with it
 *  before sending. The RGB here is what the hatch on the canvas is drawn in
 *  (see hatchTile): blue, so a mask never reads as part of the picture, which
 *  the old accent-orange tint did.
 */
export const MASK_RGB = [77, 163, 255];

/** The tile the mask is displayed with: slanted lines over a faint wash.
 *
 *  Both live in one tile so a single source-in pass over the mask raster
 *  scales them by mask strength together; a second pass would multiply the
 *  alpha twice and the lines would all but vanish on a soft edge.
 *
 *  `scale` is canvas pixels per CSS pixel. The period is fixed on screen, not
 *  on the canvas: a 2048-wide mask shown at 700px would otherwise draw its
 *  lines three screen pixels apart and moiré.
 */
export function hatchTile(scale) {
  const size = Math.max(4, Math.round(10 * scale));
  const c = document.createElement("canvas");
  c.width = size;
  c.height = size;
  const ctx = c.getContext("2d");
  const [r, g, b] = MASK_RGB;
  ctx.fillStyle = `rgba(${r},${g},${b},0.18)`;
  ctx.fillRect(0, 0, size, size);
  ctx.strokeStyle = `rgba(${r},${g},${b},0.85)`;
  ctx.lineWidth = Math.max(1, 1.2 * scale);
  ctx.lineCap = "square";
  ctx.beginPath();
  // The diagonal, plus the two corner halves, so the tile repeats seamlessly.
  ctx.moveTo(0, size);
  ctx.lineTo(size, 0);
  ctx.moveTo(-size / 2, size / 2);
  ctx.lineTo(size / 2, -size / 2);
  ctx.moveTo(size / 2, size * 1.5);
  ctx.lineTo(size * 1.5, size / 2);
  ctx.stroke();
  return c;
}

/** A freehand brush stroke. Points and size are normalised to the canvas. */
export function brushStroke({ points, size, hard, mode }) {
  return { type: "brush", mode: mode || "add", points, size, hard: !!hard };
}

/** A wand or contrast-split result.
 *
 *  Both are functions of the pixels underneath, and those change after every
 *  fill — re-evaluating on replay would make the selection drift under the
 *  user. So the rasterised result is cached on the stroke and only ever scaled.
 *  InvokeAI takes the same position: Select Object bakes its result in on Apply.
 */
export function cachedStroke(type, cache, mode) {
  return { type, mode: mode || "add", cache };
}

export function invertStroke() {
  return { type: "invert" };
}

/** Translate everything painted so far by (dx, dy), normalised to the canvas.
 *
 *  A stroke rather than a field on the mask, and that matters: an offset
 *  applied at replay time would shift every stroke painted *after* the move
 *  too, so a brush stroke would look right live and then jump on release.
 *  As a stroke it moves what came before it and leaves what comes after it
 *  where it was painted. Whatever leaves the canvas is clipped, so
 *  consecutive moves are folded together (see moveMask) and a drag off and
 *  back is a net zero.
 */
export function moveStroke(dx, dy) {
  return { type: "move", dx, dy };
}

function shiftCanvas(ctx, w, h, dx, dy) {
  if (!dx && !dy) return;
  const img = ctx.getImageData(0, 0, w, h);
  ctx.clearRect(0, 0, w, h);
  ctx.putImageData(img, dx, dy);
}

/** Wrap an alpha channel in an offscreen canvas, tinted, ready to scale. */
export function alphaToCache(alpha, w, h) {
  const c = document.createElement("canvas");
  c.width = w;
  c.height = h;
  const img = new ImageData(w, h);
  const [r, g, b] = MASK_RGB;
  for (let i = 0; i < w * h; i += 1) {
    const o = i * 4;
    img.data[o] = r;
    img.data[o + 1] = g;
    img.data[o + 2] = b;
    img.data[o + 3] = alpha[i] > 8 ? alpha[i] : 0;
  }
  c.getContext("2d").putImageData(img, 0, 0);
  return c;
}

/** Wrap an already-tinted ImageData (what buildContrastMask returns). */
export function overlayToCache(overlay, w, h) {
  const c = document.createElement("canvas");
  c.width = w;
  c.height = h;
  c.getContext("2d").putImageData(overlay, 0, 0);
  return c;
}

function brushFill(hard, erase) {
  if (erase) return hard ? "rgba(0,0,0,1)" : null;
  const [r, g, b] = MASK_RGB;
  return hard ? `rgba(${r},${g},${b},0.7)` : null;
}

function softGradient(ctx, x, y, radius, erase) {
  const g = ctx.createRadialGradient(x, y, 0, x, y, radius);
  if (erase) {
    g.addColorStop(0, "rgba(0,0,0,0.85)");
    g.addColorStop(0.55, "rgba(0,0,0,0.35)");
    g.addColorStop(1, "rgba(0,0,0,0)");
  } else {
    const [r, gg, b] = MASK_RGB;
    g.addColorStop(0, `rgba(${r},${gg},${b},0.55)`);
    g.addColorStop(0.55, `rgba(${r},${gg},${b},0.22)`);
    g.addColorStop(1, `rgba(${r},${gg},${b},0)`);
  }
  return g;
}

/** One dab of a soft brush. */
export function stampDab(ctx, x, y, radius, { hard, erase }) {
  if (hard) {
    ctx.fillStyle = brushFill(true, erase);
  } else {
    ctx.fillStyle = softGradient(ctx, x, y, radius, erase);
  }
  ctx.beginPath();
  ctx.arc(x, y, radius, 0, Math.PI * 2);
  ctx.fill();
}

/** Join two points of a hard brush with a round-capped line. */
export function strokeSegment(ctx, from, to, radius, { erase }) {
  ctx.strokeStyle = brushFill(true, erase);
  ctx.lineWidth = radius * 2;
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  ctx.beginPath();
  ctx.moveTo(from.x, from.y);
  ctx.lineTo(to.x, to.y);
  ctx.stroke();
}

/** Paint one point of a brush stroke, continuing from `prev` if there is one.
 *
 *  Shared by live painting and by replay so an undone-then-replayed stroke
 *  cannot look different from the one that was drawn.
 */
export function paintBrushPoint(ctx, prev, pt, radius, opts) {
  const { hard, erase } = opts;
  ctx.globalCompositeOperation = erase ? "destination-out" : "source-over";
  if (!prev) {
    stampDab(ctx, pt.x, pt.y, radius, opts);
    return;
  }
  if (hard) {
    strokeSegment(ctx, prev, pt, radius, opts);
    return;
  }
  const dx = pt.x - prev.x;
  const dy = pt.y - prev.y;
  const dist = Math.hypot(dx, dy);
  const step = Math.max(radius * 0.35, 1);
  const n = Math.max(1, Math.ceil(dist / step));
  for (let i = 1; i <= n; i += 1) {
    stampDab(ctx, prev.x + (dx * i) / n, prev.y + (dy * i) / n, radius, opts);
  }
}

function invertCanvas(ctx, w, h) {
  const img = ctx.getImageData(0, 0, w, h);
  const [r, g, b] = MASK_RGB;
  for (let i = 0; i < w * h; i += 1) {
    const o = i * 4;
    const a = 255 - img.data[o + 3];
    img.data[o] = r;
    img.data[o + 1] = g;
    img.data[o + 2] = b;
    img.data[o + 3] = a;
  }
  ctx.putImageData(img, 0, 0);
}

function replayBrush(ctx, stroke, w, h) {
  const erase = stroke.mode === "subtract";
  const radius = Math.max((stroke.size * w) / 2, 0.5);
  const opts = { hard: stroke.hard, erase };
  let prev = null;
  for (const p of stroke.points) {
    const pt = { x: p.x * w, y: p.y * h };
    paintBrushPoint(ctx, prev, pt, radius, opts);
    prev = pt;
  }
}

/** Replay one mask's strokes onto `target` at w x h. */
export function rasterize(mask, target, w, h) {
  if (!target || !w || !h) return;
  if (target.width !== w || target.height !== h) {
    target.width = w;
    target.height = h;
  }
  const ctx = target.getContext("2d");
  ctx.globalAlpha = 1;
  ctx.globalCompositeOperation = "source-over";
  ctx.clearRect(0, 0, w, h);
  for (const s of mask?.strokes || []) {
    if (s.type === "invert") {
      ctx.globalCompositeOperation = "source-over";
      invertCanvas(ctx, w, h);
      continue;
    }
    if (s.type === "move") {
      ctx.globalCompositeOperation = "source-over";
      shiftCanvas(ctx, w, h, Math.round(s.dx * w), Math.round(s.dy * h));
      continue;
    }
    ctx.globalCompositeOperation = s.mode === "subtract" ? "destination-out" : "source-over";
    if (s.type === "brush") replayBrush(ctx, s, w, h);
    else if (s.cache) ctx.drawImage(s.cache, 0, 0, w, h);
  }
  ctx.globalCompositeOperation = "source-over";
}
