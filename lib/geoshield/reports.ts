import { assessmentResultSchema, type AssessmentResult } from './types';

/** Serialize a result to JSON, validating it against the schema first so a
 * malformed report can never be exported. Never includes original image bytes. */
export function buildAssessmentJson(result: AssessmentResult): string {
  const validated = assessmentResultSchema.parse(result);
  return JSON.stringify(validated, null, 2);
}

function csvField(value: string | number): string {
  const text = String(value);
  if (/[",\r\n]/.test(text)) {
    return `"${text.replace(/"/g, '""')}"`;
  }
  return text;
}

const CSV_HEADER = [
  'id',
  'damageClass',
  'confidence',
  'boundingBoxX',
  'boundingBoxY',
  'boundingBoxWidth',
  'boundingBoxHeight',
  'pixelArea',
];

/** One CSV row per estimated region. A result with zero regions still
 * produces a valid CSV containing only the header row. */
export function buildRegionsCsv(result: AssessmentResult): string {
  const validated = assessmentResultSchema.parse(result);
  const rows = validated.regions.map((region) => [
    region.id,
    region.damageClass,
    region.confidence,
    ...region.normalizedBoundingBox,
    region.pixelArea,
  ]);
  return [CSV_HEADER, ...rows].map((row) => row.map(csvField).join(',')).join('\r\n') + '\r\n';
}

function sanitizeForFilename(value: string): string {
  return value.replace(/[^a-zA-Z0-9._-]+/g, '_');
}

/** e.g. geoshield-assessment-siamese-2026-08-31T20-52-00-000Z.json */
export function reportFilename(kind: 'assessment' | 'regions', result: AssessmentResult, extension: string, now: Date = new Date()): string {
  const timestamp = now.toISOString().replace(/[:.]/g, '-');
  return `geoshield-${kind}-${sanitizeForFilename(result.modelVersion)}-${timestamp}.${extension}`;
}
