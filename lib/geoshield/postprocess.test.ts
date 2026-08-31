import { describe, expect, it } from 'vitest';
import { postprocessMask } from './postprocess';

function buildProbabilities(width: number, height: number, fill: (classId: number, x: number, y: number) => number): Float32Array {
  const pixelCount = width * height;
  const probabilities = new Float32Array(pixelCount * 5);
  for (let classId = 0; classId < 5; classId++) {
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        probabilities[classId * pixelCount + y * width + x] = fill(classId, x, y);
      }
    }
  }
  return probabilities;
}

/** Uniform probability mass on whatever class the mask assigns at each pixel. */
function confidentProbabilities(mask: Uint8Array, width: number, height: number, confidence = 0.9): Float32Array {
  return buildProbabilities(width, height, (classId, x, y) => {
    const assigned = mask[y * width + x];
    if (classId === assigned) return confidence;
    return (1 - confidence) / 4;
  });
}

describe('postprocessMask', () => {
  it('returns a valid zero-result state for an empty (all-background) mask', () => {
    const width = 8;
    const height = 8;
    const mask = new Uint8Array(width * height); // all zero
    const probabilities = confidentProbabilities(mask, width, height);

    const { regions, classStatistics } = postprocessMask(mask, probabilities, width, height);

    expect(regions).toEqual([]);
    for (const stats of Object.values(classStatistics)) {
      expect(stats).toEqual({ pixelArea: 0, percentage: 0, estimatedRegions: 0 });
    }
  });

  it('detects a single isolated region above the noise floor', () => {
    const width = 10;
    const height = 10;
    const mask = new Uint8Array(width * height);
    // 5x5 = 25px block of class 3 (major), comfortably above the 20px floor.
    for (let y = 1; y <= 5; y++) {
      for (let x = 1; x <= 5; x++) {
        mask[y * width + x] = 3;
      }
    }
    const probabilities = confidentProbabilities(mask, width, height, 0.8);

    const { regions, classStatistics } = postprocessMask(mask, probabilities, width, height);

    expect(regions).toHaveLength(1);
    expect(regions[0].damageClass).toBe(3);
    expect(regions[0].pixelArea).toBe(25);
    expect(regions[0].confidence).toBeCloseTo(0.8, 5);
    expect(regions[0].normalizedBoundingBox).toEqual([0.1, 0.1, 0.5, 0.5]);
    expect(classStatistics.major.pixelArea).toBe(25);
    expect(classStatistics.major.percentage).toBe(100);
    expect(classStatistics.major.estimatedRegions).toBe(1);
  });

  it('drops connected components smaller than the noise floor and excludes them from area stats', () => {
    const width = 10;
    const height = 10;
    const mask = new Uint8Array(width * height);
    // A single noise pixel, far below the 20px minimum.
    mask[5 * width + 5] = 2;
    const probabilities = confidentProbabilities(mask, width, height);

    const { regions, classStatistics } = postprocessMask(mask, probabilities, width, height);

    expect(regions).toEqual([]);
    expect(classStatistics.minor).toEqual({ pixelArea: 0, percentage: 0, estimatedRegions: 0 });
  });

  it('merges two differently-classed blobs that touch diagonally into one region (documented 8-connectivity behavior)', () => {
    const width = 12;
    const height = 12;
    const mask = new Uint8Array(width * height);
    // A 4x4 "minor" block and a 4x4 "destroyed" block touching only at one
    // diagonal corner: under 8-connectivity these are a single component.
    for (let y = 1; y <= 4; y++) {
      for (let x = 1; x <= 4; x++) mask[y * width + x] = 2;
    }
    for (let y = 5; y <= 8; y++) {
      for (let x = 5; x <= 8; x++) mask[y * width + x] = 4;
    }
    const probabilities = confidentProbabilities(mask, width, height);

    const { regions } = postprocessMask(mask, probabilities, width, height);

    expect(regions).toHaveLength(1);
    // Equal pixel counts (16 each): the majority-class tie-break keeps
    // whichever class was encountered first by the scan (top-left block).
    expect(regions[0].pixelArea).toBe(32);
    expect(regions[0].damageClass).toBe(2);
  });

  it('assigns the majority non-background class within a mixed-class region', () => {
    const width = 10;
    const height = 10;
    const mask = new Uint8Array(width * height);
    // A 5x5 block mostly class 1, with a 2x2 pocket of class 4 inside it.
    for (let y = 1; y <= 5; y++) {
      for (let x = 1; x <= 5; x++) mask[y * width + x] = 1;
    }
    mask[2 * width + 2] = 4;
    mask[2 * width + 3] = 4;
    mask[3 * width + 2] = 4;
    mask[3 * width + 3] = 4;
    const probabilities = confidentProbabilities(mask, width, height, 0.7);

    const { regions, classStatistics } = postprocessMask(mask, probabilities, width, height);

    expect(regions).toHaveLength(1);
    expect(regions[0].damageClass).toBe(1); // 21 class-1 pixels vs 4 class-4 pixels
    expect(regions[0].pixelArea).toBe(25);
    expect(classStatistics.undamaged.estimatedRegions).toBe(1);
    expect(classStatistics.destroyed.estimatedRegions).toBe(0);
  });

  it('reports percentages that sum to 100 across multiple separate regions', () => {
    const width = 20;
    const height = 10;
    const mask = new Uint8Array(width * height);
    for (let y = 1; y <= 5; y++) for (let x = 1; x <= 5; x++) mask[y * width + x] = 1; // 25px
    for (let y = 1; y <= 5; y++) for (let x = 10; x <= 14; x++) mask[y * width + x] = 3; // 25px
    const probabilities = confidentProbabilities(mask, width, height);

    const { classStatistics } = postprocessMask(mask, probabilities, width, height);
    const total = Object.values(classStatistics).reduce((sum, stat) => sum + stat.percentage, 0);
    expect(total).toBeCloseTo(100, 5);
    expect(classStatistics.undamaged.percentage).toBeCloseTo(50, 5);
    expect(classStatistics.major.percentage).toBeCloseTo(50, 5);
  });

  it('rejects mismatched mask/probabilities dimensions', () => {
    const mask = new Uint8Array(4);
    const probabilities = new Float32Array(4 * 5 - 1);
    expect(() => postprocessMask(mask, probabilities, 2, 2)).toThrow();
  });
});
