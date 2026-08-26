/** Read design tokens for SVG code that cannot use CSS custom properties directly.
 *
 *  Charts and the UNet map used to hardcode hex values, so any palette change
 *  silently skipped them. Values are read once per call from :root.
 */
export function tokens(names) {
  const cs = typeof window !== "undefined"
    ? getComputedStyle(document.documentElement)
    : null;
  const out = {};
  for (const [key, name] of Object.entries(names)) {
    out[key] = (cs?.getPropertyValue(name) || "").trim();
  }
  return out;
}
