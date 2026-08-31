export const DAMAGE_COLORS: Record<1 | 2 | 3 | 4, [number, number, number]> = {
  1: [89, 211, 155], // undamaged — green
  2: [246, 200, 95], // minor — yellow
  3: [239, 139, 78], // major — orange
  4: [238, 101, 113], // destroyed — red
};

/** Build an RGBA buffer (suitable for `new ImageData(buffer, width, height)`
 * at the call site — kept out of this function so it stays testable without
 * a DOM) coloring each pixel by its damage class, at the given opacity, for
 * only the currently-visible classes. Background and hidden-class pixels
 * are left fully transparent. */
export function buildOverlayRgba(mask: Uint8Array, width: number, height: number, opacity: number, visibleClasses: ReadonlySet<number>): Uint8ClampedArray {
  const pixelCount = width * height;
  if (mask.length !== pixelCount) {
    throw new Error(`mask length ${mask.length} does not match width*height ${pixelCount}`);
  }
  const alpha = Math.round(Math.max(0, Math.min(1, opacity)) * 255);
  const rgba = new Uint8ClampedArray(pixelCount * 4);
  for (let i = 0; i < pixelCount; i++) {
    const classId = mask[i];
    if (classId === 0 || !visibleClasses.has(classId)) continue;
    const color = DAMAGE_COLORS[classId as 1 | 2 | 3 | 4];
    if (!color) continue;
    const base = i * 4;
    rgba[base] = color[0];
    rgba[base + 1] = color[1];
    rgba[base + 2] = color[2];
    rgba[base + 3] = alpha;
  }
  return rgba;
}
