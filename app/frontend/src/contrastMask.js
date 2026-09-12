/** Build a paint mask from luminance or local-contrast splits. */

function luminance(r, g, b) {
  return 0.2126 * r + 0.7152 * g + 0.0722 * b;
}

export function otsuThreshold(hist, total) {
  let sum = 0;
  for (let i = 0; i < 256; i += 1) sum += i * hist[i];
  let sumB = 0;
  let wB = 0;
  let maxVar = 0;
  let threshold = 128;
  for (let t = 0; t < 256; t += 1) {
    wB += hist[t];
    if (wB === 0) continue;
    const wF = total - wB;
    if (wF === 0) break;
    sumB += t * hist[t];
    const mB = sumB / wB;
    const mF = (sum - sumB) / wF;
    const varBetween = wB * wF * (mB - mF) ** 2;
    if (varBetween > maxVar) {
      maxVar = varBetween;
      threshold = t;
    }
  }
  return threshold;
}

function minMaxFloat(values) {
  let min = Infinity;
  let max = -Infinity;
  for (let i = 0; i < values.length; i += 1) {
    min = Math.min(min, values[i]);
    max = Math.max(max, values[i]);
  }
  return { min, max };
}

function otsuOnFloat(values) {
  const { min, max } = minMaxFloat(values);
  const hist = new Uint32Array(256);
  const span = Math.max(max - min, 1e-6);
  for (let i = 0; i < values.length; i += 1) {
    const bin = Math.min(255, Math.max(0, Math.round(((values[i] - min) / span) * 255)));
    hist[bin] += 1;
  }
  return min + (otsuThreshold(hist, values.length) / 255) * span;
}

function loadImageData(src) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => {
      const w = img.naturalWidth || img.width;
      const h = img.naturalHeight || img.height;
      if (!w || !h) {
        reject(new Error("Canvas image has no size"));
        return;
      }
      const c = document.createElement("canvas");
      c.width = w;
      c.height = h;
      const ctx = c.getContext("2d");
      ctx.drawImage(img, 0, 0, w, h);
      resolve({ data: ctx.getImageData(0, 0, w, h).data, w, h });
    };
    img.onerror = () => reject(new Error("Could not load canvas image"));
    img.src = src;
  });
}

function localContrastMap(luma, w, h, radius = 2) {
  const out = new Float32Array(w * h);
  for (let y = 0; y < h; y += 1) {
    for (let x = 0; x < w; x += 1) {
      let sum = 0;
      let sumSq = 0;
      let n = 0;
      for (let dy = -radius; dy <= radius; dy += 1) {
        for (let dx = -radius; dx <= radius; dx += 1) {
          const nx = x + dx;
          const ny = y + dy;
          if (nx < 0 || ny < 0 || nx >= w || ny >= h) continue;
          const v = luma[ny * w + nx];
          sum += v;
          sumSq += v * v;
          n += 1;
        }
      }
      const mean = sum / n;
      out[y * w + x] = Math.sqrt(Math.max(0, sumSq / n - mean * mean));
    }
  }
  return out;
}

function blurAlpha(alpha, w, h, radius) {
  if (radius <= 0) return alpha;
  const out = new Uint8ClampedArray(alpha.length);
  for (let y = 0; y < h; y += 1) {
    for (let x = 0; x < w; x += 1) {
      let sum = 0;
      let n = 0;
      for (let dy = -radius; dy <= radius; dy += 1) {
        for (let dx = -radius; dx <= radius; dx += 1) {
          const nx = x + dx;
          const ny = y + dy;
          if (nx < 0 || ny < 0 || nx >= w || ny >= h) continue;
          sum += alpha[ny * w + nx];
          n += 1;
        }
      }
      out[y * w + x] = Math.round(sum / n);
    }
  }
  return out;
}

function blurFloat(values, w, h, radius) {
  if (radius <= 0) return values;
  const out = new Float32Array(values.length);
  for (let y = 0; y < h; y += 1) {
    for (let x = 0; x < w; x += 1) {
      let sum = 0;
      let n = 0;
      for (let dy = -radius; dy <= radius; dy += 1) {
        for (let dx = -radius; dx <= radius; dx += 1) {
          const nx = x + dx;
          const ny = y + dy;
          if (nx < 0 || ny < 0 || nx >= w || ny >= h) continue;
          sum += values[ny * w + nx];
          n += 1;
        }
      }
      out[y * w + x] = sum / n;
    }
  }
  return out;
}

/** Shared split logic for preview tiles and mask overlay. */
export function computeContrastSplit(data, w, h, opts = {}) {
  const {
    method = "luminance",
    region = "dark",
    autoThreshold = true,
    threshold = 128,
    thresholdBias = 0,
    soften = 0,
    presmooth = 0,
  } = opts;

  const n = w * h;
  let luma = new Float32Array(n);

  for (let i = 0; i < n; i += 1) {
    const o = i * 4;
    luma[i] = luminance(data[o], data[o + 1], data[o + 2]);
  }

  if (presmooth > 0) {
    luma = blurFloat(luma, w, h, Math.round(presmooth));
  }

  const hist = new Uint32Array(256);
  for (let i = 0; i < n; i += 1) {
    hist[Math.min(255, Math.max(0, Math.round(luma[i])))] += 1;
  }

  let values = luma;
  let cut;
  if (method === "contrast") {
    values = localContrastMap(luma, w, h, 2);
    cut = autoThreshold !== false ? otsuOnFloat(values) : threshold;
    if (autoThreshold !== false) {
      const { min, max } = minMaxFloat(values);
      cut += (thresholdBias || 0) * (max - min) / 512;
    }
  } else if (autoThreshold === false) {
    cut = Math.min(255, Math.max(1, threshold ?? 128));
  } else {
    cut = Math.min(255, Math.max(1, otsuThreshold(hist, n) + (thresholdBias || 0)));
  }

  const maskAlpha = new Uint8ClampedArray(n);
  for (let i = 0; i < n; i += 1) {
    let on;
    if (method === "contrast") {
      const high = values[i] >= cut;
      on = region === "high" ? high : region === "low" ? !high : high;
    } else {
      const dark = luma[i] <= cut;
      on = region === "dark" ? dark : region === "light" ? !dark : dark;
    }
    maskAlpha[i] = on ? 255 : 0;
  }

  const softened = soften > 0 ? blurAlpha(maskAlpha, w, h, soften) : maskAlpha;
  return { luma, values, cut, maskAlpha: softened, w, h };
}

export async function detectBrightnessThreshold(imageSrc) {
  const { data, w, h } = await loadImageData(imageSrc);
  const hist = new Uint32Array(256);
  for (let i = 0; i < w * h; i += 1) {
    const o = i * 4;
    const y = luminance(data[o], data[o + 1], data[o + 2]);
    hist[Math.min(255, Math.max(0, Math.round(y)))] += 1;
  }
  const t = otsuThreshold(hist, w * h);
  return Math.max(1, Math.min(255, t));
}

/**
 * @param {string} imageSrc data URL of the canvas image
 * @param {object} opts
 * @param {'luminance'|'contrast'} opts.method
 * @param {'dark'|'light'|'high'|'low'} opts.region - which side becomes the mask
 * @param {boolean} opts.autoThreshold
 * @param {number} opts.threshold 1-255 manual luminance cutoff
 * @param {number} opts.thresholdBias -128..128 applied after auto detect
 * @param {number} opts.soften 0-8px edge soften
 */
export async function buildContrastMask(imageSrc, opts = {}) {
  const { data, w, h } = await loadImageData(imageSrc);
  const { cut, maskAlpha } = computeContrastSplit(data, w, h, opts);
  const n = w * h;
  const overlay = new ImageData(w, h);
  for (let i = 0; i < n; i += 1) {
    const o = i * 4;
    const a = maskAlpha[i];
    overlay.data[o] = 255;
    overlay.data[o + 1] = 122;
    overlay.data[o + 2] = 69;
    overlay.data[o + 3] = a > 8 ? a : 0;
  }
  return { overlay, w, h, cut: Math.round(cut * 10) / 10 };
}

/** Preview tiles for fg/bg pickers (grayscale, same size as thumb). */
export async function contrastPreview(imageSrc, opts, size = 96) {
  const { data, w, h } = await loadImageData(imageSrc);
  const { values, cut, luma } = computeContrastSplit(data, w, h, { ...opts, region: "dark", soften: 0 });
  const n = w * h;

  const fg = document.createElement("canvas");
  const bg = document.createElement("canvas");
  fg.width = bg.width = w;
  fg.height = bg.height = h;
  const fgCtx = fg.getContext("2d");
  const bgCtx = bg.getContext("2d");
  const fgImg = fgCtx.createImageData(w, h);
  const bgImg = bgCtx.createImageData(w, h);

  for (let i = 0; i < n; i += 1) {
    const o = i * 4;
    let fgOn;
    if (opts.method === "contrast") {
      fgOn = values[i] >= cut;
    } else {
      fgOn = luma[i] <= cut;
    }
    if (fgOn) {
      fgImg.data[o] = data[o];
      fgImg.data[o + 1] = data[o + 1];
      fgImg.data[o + 2] = data[o + 2];
      fgImg.data[o + 3] = 255;
      bgImg.data[o + 3] = 255;
      bgImg.data[o] = bgImg.data[o + 1] = bgImg.data[o + 2] = 24;
    } else {
      bgImg.data[o] = data[o];
      bgImg.data[o + 1] = data[o + 1];
      bgImg.data[o + 2] = data[o + 2];
      bgImg.data[o + 3] = 255;
      fgImg.data[o + 3] = 255;
      fgImg.data[o] = fgImg.data[o + 1] = fgImg.data[o + 2] = 24;
    }
  }
  fgCtx.putImageData(fgImg, 0, 0);
  bgCtx.putImageData(bgImg, 0, 0);

  const thumb = (c) => {
    const t = document.createElement("canvas");
    t.width = t.height = size;
    const ctx = t.getContext("2d");
    const scale = Math.min(size / w, size / h);
    const dw = w * scale;
    const dh = h * scale;
    ctx.fillStyle = "#15171e";
    ctx.fillRect(0, 0, size, size);
    ctx.drawImage(c, (size - dw) / 2, (size - dh) / 2, dw, dh);
    return t.toDataURL("image/png");
  };

  return { foreground: thumb(fg), background: thumb(bg), cut: Math.round(cut * 10) / 10 };
}

/** Blend processed image inside mask onto original outside mask. All inputs are data URLs. */
export async function compositePostprocWithMask(baseSrc, processedSrc, maskDataUrl) {
  const [base, processed, mask] = await Promise.all([
    loadImageData(baseSrc),
    loadImageData(processedSrc),
    loadImageData(maskDataUrl),
  ]);
  const w = base.w;
  const h = base.h;
  if (processed.w !== w || processed.h !== h || mask.w !== w || mask.h !== h) {
    return processedSrc;
  }
  const out = new Uint8ClampedArray(base.data.length);
  for (let i = 0; i < w * h; i += 1) {
    const o = i * 4;
    const ma = mask.data[o + 3] / 255;
    if (ma <= 0.02) {
      out[o] = base.data[o];
      out[o + 1] = base.data[o + 1];
      out[o + 2] = base.data[o + 2];
      out[o + 3] = 255;
    } else if (ma >= 0.98) {
      out[o] = processed.data[o];
      out[o + 1] = processed.data[o + 1];
      out[o + 2] = processed.data[o + 2];
      out[o + 3] = 255;
    } else {
      out[o] = Math.round(base.data[o] * (1 - ma) + processed.data[o] * ma);
      out[o + 1] = Math.round(base.data[o + 1] * (1 - ma) + processed.data[o + 1] * ma);
      out[o + 2] = Math.round(base.data[o + 2] * (1 - ma) + processed.data[o + 2] * ma);
      out[o + 3] = 255;
    }
  }
  const c = document.createElement("canvas");
  c.width = w;
  c.height = h;
  c.getContext("2d").putImageData(new ImageData(out, w, h), 0, 0);
  return c.toDataURL("image/png");
}

/** Painted pixels of a mask canvas: how many, and the box around them.
 *
 *  One pass gives both, so the bounding box costs nothing over the count that
 *  was already being taken. `bbox` is in canvas pixels, x1/y1 exclusive, null
 *  when nothing is painted.
 */
export function measureMask(maskCanvas) {
  if (!maskCanvas?.width || !maskCanvas?.height) return { count: 0, bbox: null };
  const w = maskCanvas.width;
  const h = maskCanvas.height;
  const { data } = maskCanvas.getContext("2d").getImageData(0, 0, w, h);
  let count = 0;
  let x0 = w;
  let y0 = h;
  let x1 = -1;
  let y1 = -1;
  for (let y = 0; y < h; y += 1) {
    let i = y * w * 4 + 3;
    for (let x = 0; x < w; x += 1, i += 4) {
      if (data[i] > 8) {
        count += 1;
        if (x < x0) x0 = x;
        if (x > x1) x1 = x;
        if (y < y0) y0 = y;
        if (y > y1) y1 = y;
      }
    }
  }
  return { count, bbox: count ? { x0, y0, x1: x1 + 1, y1: y1 + 1 } : null };
}

/** Count painted mask pixels from a mask canvas element. */
export const countMaskPixels = (maskCanvas) => measureMask(maskCanvas).count;

/**
 * Contiguous flood fill from a seed pixel (magic wand).
 * @param {string} imageSrc data URL
 * @param {{ x: number, y: number, tolerance?: number, soften?: number }} opts
 *   tolerance is 0-100 (mapped to 0-255 max-channel delta).
 * @returns {Promise<{ alpha: Uint8ClampedArray, w: number, h: number }>}
 */
export async function floodFillMask(imageSrc, opts = {}) {
  const { data, w, h } = await loadImageData(imageSrc);
  const sx = Math.max(0, Math.min(w - 1, Math.round(opts.x ?? 0)));
  const sy = Math.max(0, Math.min(h - 1, Math.round(opts.y ?? 0)));
  const tolSlider = Math.max(0, Math.min(100, opts.tolerance ?? 30));
  const tol = Math.round((tolSlider / 100) * 255);
  const soften = Math.max(0, Math.round(opts.soften ?? 0));

  const seedOff = (sy * w + sx) * 4;
  const sr = data[seedOff];
  const sg = data[seedOff + 1];
  const sb = data[seedOff + 2];

  const n = w * h;
  const visited = new Uint8Array(n);
  const alpha = new Uint8ClampedArray(n);
  const stack = new Int32Array(n);
  let top = 0;
  stack[top++] = sy * w + sx;

  const matches = (i) => {
    const o = i * 4;
    const dr = Math.abs(data[o] - sr);
    const dg = Math.abs(data[o + 1] - sg);
    const db = Math.abs(data[o + 2] - sb);
    return Math.max(dr, dg, db) <= tol;
  };

  while (top > 0) {
    const idx = stack[--top];
    if (visited[idx]) continue;
    if (!matches(idx)) continue;
    visited[idx] = 1;
    alpha[idx] = 255;

    const x = idx % w;
    const y = (idx - x) / w;
    if (x > 0) stack[top++] = idx - 1;
    if (x < w - 1) stack[top++] = idx + 1;
    if (y > 0) stack[top++] = idx - w;
    if (y < h - 1) stack[top++] = idx + w;
  }

  const softened = soften > 0 ? blurAlpha(alpha, w, h, soften) : alpha;
  return { alpha: softened, w, h };
}
