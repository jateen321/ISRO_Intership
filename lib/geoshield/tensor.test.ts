import { describe, expect, it } from 'vitest';
import {
  IMAGENET_MEAN,
  IMAGENET_STD,
  normalizeImageNet,
  softmaxArgmax,
  splitIntoTiles,
  stitchTiles,
  validateImagePair,
} from './tensor';

describe('validateImagePair', () => {
  it('accepts a matching square pair in range', () => {
    expect(validateImagePair({ width: 1024, height: 1024 }, { width: 1024, height: 1024 })).toEqual({ ok: true });
  });

  it('rejects a non-square image', () => {
    const result = validateImagePair({ width: 1024, height: 768 }, { width: 1024, height: 768 });
    expect(result.ok).toBe(false);
    expect(result.error).toMatch(/square/i);
  });

  it('rejects mismatched before/after dimensions', () => {
    const result = validateImagePair({ width: 1024, height: 1024 }, { width: 2048, height: 2048 });
    expect(result.ok).toBe(false);
    expect(result.error).toMatch(/identical dimensions/i);
  });

  it('rejects images below the minimum size', () => {
    const result = validateImagePair({ width: 256, height: 256 }, { width: 256, height: 256 });
    expect(result.ok).toBe(false);
    expect(result.error).toMatch(/between/i);
  });

  it('rejects images above the maximum size', () => {
    const result = validateImagePair({ width: 8192, height: 8192 }, { width: 8192, height: 8192 });
    expect(result.ok).toBe(false);
  });
});

describe('normalizeImageNet', () => {
  it('applies per-channel ImageNet mean/std normalization', () => {
    // A single opaque red-ish pixel.
    const rgba = new Uint8ClampedArray([128, 64, 32, 255]);
    const chw = normalizeImageNet(rgba, 1, 1);
    expect(chw).toHaveLength(3);
    expect(chw[0]).toBeCloseTo((128 / 255 - IMAGENET_MEAN[0]) / IMAGENET_STD[0], 6);
    expect(chw[1]).toBeCloseTo((64 / 255 - IMAGENET_MEAN[1]) / IMAGENET_STD[1], 6);
    expect(chw[2]).toBeCloseTo((32 / 255 - IMAGENET_MEAN[2]) / IMAGENET_STD[2], 6);
  });

  it('rejects a length mismatch', () => {
    expect(() => normalizeImageNet(new Uint8ClampedArray(3), 1, 1)).toThrow();
  });
});

describe('splitIntoTiles / stitchTiles', () => {
  it('round-trips a float tensor through split then stitch', () => {
    const channels = 3;
    const fullSize = 8;
    const tileSize = 4;
    const data = new Float32Array(channels * fullSize * fullSize);
    for (let i = 0; i < data.length; i++) data[i] = i;

    const tiles = splitIntoTiles(data, channels, fullSize, tileSize, Float32Array);
    expect(tiles).toHaveLength(4);
    expect(tiles[0]).toHaveLength(channels * tileSize * tileSize);

    const stitched = stitchTiles(tiles, channels, fullSize, tileSize, Float32Array);
    expect(Array.from(stitched)).toEqual(Array.from(data));
  });

  it('round-trips a uint8 mask through split then stitch', () => {
    const fullSize = 8;
    const tileSize = 4;
    const data = new Uint8Array(fullSize * fullSize);
    for (let i = 0; i < data.length; i++) data[i] = i % 5;

    const tiles = splitIntoTiles(data, 1, fullSize, tileSize, Uint8Array);
    const stitched = stitchTiles(tiles, 1, fullSize, tileSize, Uint8Array);
    expect(Array.from(stitched)).toEqual(Array.from(data));
  });

  it('extracts the correct quadrant for each tile', () => {
    const fullSize = 4;
    const tileSize = 2;
    // 4x4 grid, values equal to their flat index, single channel.
    const data = new Uint8Array(fullSize * fullSize);
    for (let i = 0; i < data.length; i++) data[i] = i;

    const [topLeft, topRight, bottomLeft, bottomRight] = splitIntoTiles(data, 1, fullSize, tileSize, Uint8Array);
    expect(Array.from(topLeft)).toEqual([0, 1, 4, 5]);
    expect(Array.from(topRight)).toEqual([2, 3, 6, 7]);
    expect(Array.from(bottomLeft)).toEqual([8, 9, 12, 13]);
    expect(Array.from(bottomRight)).toEqual([10, 11, 14, 15]);
  });

  it('rejects a tile count mismatch on stitch', () => {
    expect(() => stitchTiles([new Uint8Array(4)], 1, 4, 2, Uint8Array)).toThrow();
  });
});

describe('softmaxArgmax', () => {
  it('picks the highest-logit class per pixel and returns valid probabilities', () => {
    const numClasses = 3;
    const height = 1;
    const width = 2;
    // Pixel 0: class 2 wins. Pixel 1: class 0 wins.
    const logits = new Float32Array([
      /* class0 */ 0, 5,
      /* class1 */ 1, 1,
      /* class2 */ 10, -1,
    ]);
    const { mask, probabilities } = softmaxArgmax(logits, numClasses, height, width);
    expect(Array.from(mask)).toEqual([2, 0]);
    // Probabilities for each pixel across classes must sum to ~1.
    for (let pixel = 0; pixel < 2; pixel++) {
      let sum = 0;
      for (let c = 0; c < numClasses; c++) sum += probabilities[c * 2 + pixel];
      expect(sum).toBeCloseTo(1, 5);
    }
  });

  it('is numerically stable for large logits', () => {
    const logits = new Float32Array([1000, -1000, 500, -500]);
    const { mask, probabilities } = softmaxArgmax(logits, 2, 1, 2);
    expect(Array.from(mask)).toEqual([0, 1]);
    expect(probabilities.every((value) => Number.isFinite(value))).toBe(true);
  });

  it('rejects a logits length mismatch', () => {
    expect(() => softmaxArgmax(new Float32Array(5), 5, 2, 2)).toThrow();
  });
});
