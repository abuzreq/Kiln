/** The selection entity: an area of the canvas the user owns.
 *
 *  A selection used to be whatever pixels happened to be sitting on the mask
 *  canvas at the moment Generate was pressed. That made it impossible to undo
 *  precisely, to invert, to disable without losing, or to survive a resize —
 *  and it was silently thrown away after every fill.
 *
 *  Here it is a list of strokes. The mask canvas is a cache of replaying them,
 *  so undo is "drop the last stroke and replay", and a resize is "replay at the
 *  new size". Coordinates are normalised 0..1 so neither depends on the pixel
 *  dimensions the strokes happened to be drawn at.
 */

/** The overlay tint. Matches .mask-overlay in styles.css. */
export const MASK_RGB = [255, 122, 69];

const DEFAULT_PARAMS = {
  change: 0.65,
  feather: 8,
  harmonize: 2,
  bendPreset: "",
};

let counter = 0;

export function newSelection(name) {
  counter += 1;
  return {
    id: Math.random().toString(36).slice(2),
    name: name || `Selection ${counter}`,
    enabled: true,
    strokes: [],
    params: { ...DEFAULT_PARAMS },
  };
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

/** Replay a selection's strokes onto `target` at w x h. */
export function rasterize(selection, target, w, h) {
  if (!target || !w || !h) return;
  if (target.width !== w || target.height !== h) {
    target.width = w;
    target.height = h;
  }
  const ctx = target.getContext("2d");
  ctx.globalAlpha = 1;
  ctx.globalCompositeOperation = "source-over";
  ctx.clearRect(0, 0, w, h);
  for (const s of selection?.strokes || []) {
    if (s.type === "invert") {
      ctx.globalCompositeOperation = "source-over";
      invertCanvas(ctx, w, h);
      continue;
    }
    ctx.globalCompositeOperation = s.mode === "subtract" ? "destination-out" : "source-over";
    if (s.type === "brush") replayBrush(ctx, s, w, h);
    else if (s.cache) ctx.drawImage(s.cache, 0, 0, w, h);
  }
  ctx.globalCompositeOperation = "source-over";
}

/** Has anything been drawn at all? Cheaper than counting pixels. */
export function hasStrokes(selection) {
  return !!selection?.strokes?.length;
}
