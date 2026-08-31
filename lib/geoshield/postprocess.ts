import type { ClassStatistic, DamageClass, DamageRegion } from './types';

const NEIGHBOR_OFFSETS: ReadonlyArray<readonly [number, number]> = [
  [-1, -1], [0, -1], [1, -1],
  [-1, 0], [1, 0],
  [-1, 1], [0, 1], [1, 1],
];

const DEFAULT_MIN_REGION_PIXELS = 20;

export interface PostprocessOptions {
  /** Connected components with fewer pixels than this are treated as noise and dropped. */
  minRegionPixels?: number;
}

export interface PostprocessOutput {
  regions: DamageRegion[];
  classStatistics: Record<string, ClassStatistic>;
}

/**
 * Group a segmentation mask into estimated building regions and per-class
 * statistics. `mask` holds a class id (0 = background, 1-4 = damage) per
 * pixel; `probabilities` holds per-pixel softmax scores as 5 channel-major
 * planes ([class][y][x], flattened) so region confidence reflects the
 * model's own certainty rather than a fixed value.
 */
export function postprocessMask(
  mask: Uint8Array,
  probabilities: Float32Array,
  width: number,
  height: number,
  options: PostprocessOptions = {},
): PostprocessOutput {
  const minRegionPixels = options.minRegionPixels ?? DEFAULT_MIN_REGION_PIXELS;
  const pixelCount = width * height;
  if (mask.length !== pixelCount) {
    throw new Error(`mask length ${mask.length} does not match width*height ${pixelCount}`);
  }
  if (probabilities.length !== pixelCount * 5) {
    throw new Error(`probabilities length ${probabilities.length} does not match 5*width*height ${pixelCount * 5}`);
  }

  const labels = new Int32Array(pixelCount).fill(-1);
  const regions: DamageRegion[] = [];
  const queue = new Int32Array(pixelCount);

  for (let start = 0; start < pixelCount; start++) {
    if (mask[start] === 0 || labels[start] !== -1) continue;

    let queueHead = 0;
    let queueTail = 0;
    queue[queueTail++] = start;
    labels[start] = start;

    const classCounts = new Map<number, number>();
    let minX = width;
    let maxX = -1;
    let minY = height;
    let maxY = -1;
    let pixelArea = 0;
    const memberPixels: number[] = [];

    while (queueHead < queueTail) {
      const index = queue[queueHead++];
      const x = index % width;
      const y = (index - x) / width;
      pixelArea++;
      memberPixels.push(index);
      const classId = mask[index];
      classCounts.set(classId, (classCounts.get(classId) ?? 0) + 1);
      if (x < minX) minX = x;
      if (x > maxX) maxX = x;
      if (y < minY) minY = y;
      if (y > maxY) maxY = y;

      for (const [dx, dy] of NEIGHBOR_OFFSETS) {
        const nx = x + dx;
        const ny = y + dy;
        if (nx < 0 || nx >= width || ny < 0 || ny >= height) continue;
        const neighborIndex = ny * width + nx;
        if (mask[neighborIndex] === 0 || labels[neighborIndex] !== -1) continue;
        labels[neighborIndex] = start;
        queue[queueTail++] = neighborIndex;
      }
    }

    if (pixelArea < minRegionPixels) continue;

    let majorityClass = 1;
    let majorityCount = -1;
    for (const [classId, count] of classCounts) {
      if (count > majorityCount) {
        majorityCount = count;
        majorityClass = classId;
      }
    }

    let probabilitySum = 0;
    for (const index of memberPixels) {
      probabilitySum += probabilities[majorityClass * pixelCount + index];
    }

    regions.push({
      id: `region-${regions.length + 1}`,
      damageClass: majorityClass as Exclude<DamageClass, 0>,
      confidence: probabilitySum / pixelArea,
      normalizedBoundingBox: [minX / width, minY / height, (maxX - minX + 1) / width, (maxY - minY + 1) / height],
      pixelArea,
    });
  }

  const classStatistics: Record<string, ClassStatistic> = {
    undamaged: { pixelArea: 0, percentage: 0, estimatedRegions: 0 },
    minor: { pixelArea: 0, percentage: 0, estimatedRegions: 0 },
    major: { pixelArea: 0, percentage: 0, estimatedRegions: 0 },
    destroyed: { pixelArea: 0, percentage: 0, estimatedRegions: 0 },
  };
  const classKeyByValue: Record<number, string> = { 1: 'undamaged', 2: 'minor', 3: 'major', 4: 'destroyed' };

  let totalBuildingArea = 0;
  for (const region of regions) {
    const key = classKeyByValue[region.damageClass];
    classStatistics[key].pixelArea += region.pixelArea;
    classStatistics[key].estimatedRegions += 1;
    totalBuildingArea += region.pixelArea;
  }
  if (totalBuildingArea > 0) {
    for (const key of Object.keys(classStatistics)) {
      classStatistics[key].percentage = (classStatistics[key].pixelArea / totalBuildingArea) * 100;
    }
  }

  return { regions, classStatistics };
}
