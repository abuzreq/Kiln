// A saved preset's picture: the render it was saved from, cropped square and
// shrunk to a small JPEG data URL. It lives in the preset's own JSON file, so
// it goes wherever the preset does and needs no file of its own.

const SIZE = 96;

/** `src` as a preset thumbnail, or null when it cannot be read. */
export async function presetThumb(src) {
  if (!src) return null;
  try {
    const img = new Image();
    img.src = src;
    await img.decode();
    const w = img.naturalWidth;
    const h = img.naturalHeight;
    const s = Math.min(w, h);
    const c = document.createElement("canvas");
    c.width = SIZE;
    c.height = SIZE;
    c.getContext("2d").drawImage(img, (w - s) / 2, (h - s) / 2, s, s, 0, 0, SIZE, SIZE);
    return c.toDataURL("image/jpeg", 0.8);
  } catch {
    return null;
  }
}
