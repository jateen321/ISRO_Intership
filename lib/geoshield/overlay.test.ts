import { describe, expect, it } from 'vitest';
import { buildOverlayRgba, DAMAGE_COLORS } from './overlay';

describe('buildOverlayRgba', () => {
  it('colors each visible-class pixel and leaves background transparent', () => {
    const mask = new Uint8Array([0, 1, 2, 4]);
    const rgba = buildOverlayRgba(mask, 4, 1, 1, new Set([1, 2, 3, 4]));

    expect(Array.from(rgba.slice(0, 4))).toEqual([0, 0, 0, 0]); // background
    expect(Array.from(rgba.slice(4, 8))).toEqual([...DAMAGE_COLORS[1], 255]);
    expect(Array.from(rgba.slice(8, 12))).toEqual([...DAMAGE_COLORS[2], 255]);
    expect(Array.from(rgba.slice(12, 16))).toEqual([...DAMAGE_COLORS[4], 255]);
  });

  it('leaves hidden classes transparent even if non-background', () => {
    const mask = new Uint8Array([3]);
    const rgba = buildOverlayRgba(mask, 1, 1, 1, new Set([1, 2])); // class 3 not in the visible set
    expect(Array.from(rgba)).toEqual([0, 0, 0, 0]);
  });

  it('clamps opacity into 0-255 alpha', () => {
    const mask = new Uint8Array([1]);
    const over = buildOverlayRgba(mask, 1, 1, 1.5, new Set([1]));
    const under = buildOverlayRgba(mask, 1, 1, -0.5, new Set([1]));
    expect(over[3]).toBe(255);
    expect(under[3]).toBe(0);
  });

  it('rejects a mask/dimension mismatch', () => {
    expect(() => buildOverlayRgba(new Uint8Array(3), 2, 2, 1, new Set([1]))).toThrow();
  });
});
