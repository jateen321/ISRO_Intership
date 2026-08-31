export const IMAGENET_MEAN = [0.485, 0.456, 0.406] as const;
export const IMAGENET_STD = [0.229, 0.224, 0.225] as const;

export const WORKING_SIZE = 1024;
export const TILE_SIZE = 512;
export const MIN_INPUT_SIZE = 512;
export const MAX_INPUT_SIZE = 4096;

export interface ValidationResult {
  ok: boolean;
  error?: string;
}

/** Both images must be square, RGB-decodable, matching dimensions, and
 * within the size range the pipeline is built for. */
export function validateImagePair(before: { width: number; height: number }, after: { width: number; height: number }): ValidationResult {
  for (const [label, dims] of [['before', before], ['after', after]] as const) {
    if (dims.width !== dims.height) {
      return { ok: false, error: `The ${label} image must be square (got ${dims.width}x${dims.height}).` };
    }
    if (dims.width < MIN_INPUT_SIZE || dims.width > MAX_INPUT_SIZE) {
      return { ok: false, error: `The ${label} image must be between ${MIN_INPUT_SIZE} and ${MAX_INPUT_SIZE}px (got ${dims.width}px).` };
    }
  }
  if (before.width !== after.width || before.height !== after.height) {
    return { ok: false, error: 'The before and after images must have identical dimensions.' };
  }
  return { ok: true };
}

/** RGBA (from canvas ImageData, values 0-255) -> ImageNet-normalized CHW float32. */
export function normalizeImageNet(rgba: Uint8ClampedArray, width: number, height: number): Float32Array {
  const pixelCount = width * height;
  if (rgba.length !== pixelCount * 4) {
    throw new Error(`rgba length ${rgba.length} does not match width*height*4 (${pixelCount * 4})`);
  }
  const chw = new Float32Array(pixelCount * 3);
  for (let i = 0; i < pixelCount; i++) {
    const base = i * 4;
    chw[i] = (rgba[base] / 255 - IMAGENET_MEAN[0]) / IMAGENET_STD[0];
    chw[pixelCount + i] = (rgba[base + 1] / 255 - IMAGENET_MEAN[1]) / IMAGENET_STD[1];
    chw[pixelCount * 2 + i] = (rgba[base + 2] / 255 - IMAGENET_MEAN[2]) / IMAGENET_STD[2];
  }
  return chw;
}

type TypedArrayConstructor<T> = { new (length: number): T };

function tileOffsets(fullSize: number, tileSize: number): Array<[number, number]> {
  const perSide = fullSize / tileSize;
  if (!Number.isInteger(perSide)) {
    throw new Error(`fullSize ${fullSize} is not an integer multiple of tileSize ${tileSize}`);
  }
  const offsets: Array<[number, number]> = [];
  for (let tileY = 0; tileY < perSide; tileY++) {
    for (let tileX = 0; tileX < perSide; tileX++) {
      offsets.push([tileX * tileSize, tileY * tileSize]);
    }
  }
  return offsets;
}

/** Split a channel-major (C,H,W) array into equal square tiles, in
 * row-major tile order (top-left, top-right, bottom-left, bottom-right for
 * a 2x2 split). Works for both float32 image tensors and, with channels=1,
 * uint8 class masks. */
export function splitIntoTiles<T extends Float32Array | Uint8Array>(
  data: T,
  channels: number,
  fullSize: number,
  tileSize: number,
  ArrayCtor: TypedArrayConstructor<T>,
): T[] {
  const fullPlane = fullSize * fullSize;
  const tilePlane = tileSize * tileSize;
  return tileOffsets(fullSize, tileSize).map(([offsetX, offsetY]) => {
    const tile = new ArrayCtor(channels * tilePlane);
    for (let c = 0; c < channels; c++) {
      for (let y = 0; y < tileSize; y++) {
        const srcStart = c * fullPlane + (offsetY + y) * fullSize + offsetX;
        const dstStart = c * tilePlane + y * tileSize;
        tile.set(data.subarray(srcStart, srcStart + tileSize) as T, dstStart);
      }
    }
    return tile;
  });
}

/** Inverse of splitIntoTiles: reassemble tiles (in the same row-major tile
 * order) back into one channel-major (C,H,W) array. */
export function stitchTiles<T extends Float32Array | Uint8Array>(
  tiles: T[],
  channels: number,
  fullSize: number,
  tileSize: number,
  ArrayCtor: TypedArrayConstructor<T>,
): T {
  const offsets = tileOffsets(fullSize, tileSize);
  if (tiles.length !== offsets.length) {
    throw new Error(`expected ${offsets.length} tiles, got ${tiles.length}`);
  }
  const fullPlane = fullSize * fullSize;
  const tilePlane = tileSize * tileSize;
  const full = new ArrayCtor(channels * fullPlane);
  offsets.forEach(([offsetX, offsetY], tileIndex) => {
    const tile = tiles[tileIndex];
    for (let c = 0; c < channels; c++) {
      for (let y = 0; y < tileSize; y++) {
        const srcStart = c * tilePlane + y * tileSize;
        const dstStart = c * fullPlane + (offsetY + y) * fullSize + offsetX;
        full.set(tile.subarray(srcStart, srcStart + tileSize) as T, dstStart);
      }
    }
  });
  return full;
}

/** Channel-major (numClasses,H,W) logits -> per-pixel softmax probabilities
 * (same layout) and the argmax class mask. */
export function softmaxArgmax(logits: Float32Array, numClasses: number, height: number, width: number): { mask: Uint8Array; probabilities: Float32Array } {
  const pixelCount = height * width;
  if (logits.length !== numClasses * pixelCount) {
    throw new Error(`logits length ${logits.length} does not match numClasses*height*width (${numClasses * pixelCount})`);
  }
  const mask = new Uint8Array(pixelCount);
  const probabilities = new Float32Array(logits.length);
  for (let pixel = 0; pixel < pixelCount; pixel++) {
    let maxLogit = -Infinity;
    for (let c = 0; c < numClasses; c++) {
      const value = logits[c * pixelCount + pixel];
      if (value > maxLogit) maxLogit = value;
    }
    let sumExp = 0;
    for (let c = 0; c < numClasses; c++) {
      const expValue = Math.exp(logits[c * pixelCount + pixel] - maxLogit);
      probabilities[c * pixelCount + pixel] = expValue;
      sumExp += expValue;
    }
    let bestClass = 0;
    let bestProbability = -Infinity;
    for (let c = 0; c < numClasses; c++) {
      const probability = probabilities[c * pixelCount + pixel] / sumExp;
      probabilities[c * pixelCount + pixel] = probability;
      if (probability > bestProbability) {
        bestProbability = probability;
        bestClass = c;
      }
    }
    mask[pixel] = bestClass;
  }
  return { mask, probabilities };
}
