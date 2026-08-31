import { z } from 'zod';

export type DamageClass = 0 | 1 | 2 | 3 | 4;

export const DAMAGE_CLASS_NAMES = ['background', 'undamaged', 'minor', 'major', 'destroyed'] as const;

export interface DamageRegion {
  id: string;
  damageClass: Exclude<DamageClass, 0>;
  confidence: number;
  normalizedBoundingBox: [number, number, number, number];
  pixelArea: number;
}

export interface ClassStatistic {
  pixelArea: number;
  percentage: number;
  estimatedRegions: number;
}

export interface AssessmentResult {
  schemaVersion: '1.0';
  modelVersion: string;
  modelHash: string;
  runtime: 'webgpu' | 'wasm';
  inputDimensions: [number, number];
  processingTimeMs: number;
  classStatistics: Record<string, ClassStatistic>;
  regions: DamageRegion[];
  warnings: string[];
}

const damageRegionSchema = z.object({
  id: z.string().min(1),
  damageClass: z.union([z.literal(1), z.literal(2), z.literal(3), z.literal(4)]),
  confidence: z.number().min(0).max(1),
  normalizedBoundingBox: z.tuple([z.number(), z.number(), z.number(), z.number()]),
  pixelArea: z.number().nonnegative(),
});

const classStatisticSchema = z.object({
  pixelArea: z.number().nonnegative(),
  percentage: z.number().min(0).max(100),
  estimatedRegions: z.number().nonnegative(),
});

export const assessmentResultSchema = z.object({
  schemaVersion: z.literal('1.0'),
  modelVersion: z.string().min(1),
  modelHash: z.string().min(1),
  runtime: z.union([z.literal('webgpu'), z.literal('wasm')]),
  inputDimensions: z.tuple([z.number().positive(), z.number().positive()]),
  processingTimeMs: z.number().nonnegative(),
  classStatistics: z.record(z.string(), classStatisticSchema),
  regions: z.array(damageRegionSchema),
  warnings: z.array(z.string()),
});
