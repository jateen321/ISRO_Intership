import { describe, expect, it } from 'vitest';
import { buildAssessmentJson, buildRegionsCsv, reportFilename } from './reports';
import { assessmentResultSchema, type AssessmentResult } from './types';

const baseResult: AssessmentResult = {
  schemaVersion: '1.0',
  modelVersion: 'siamese-smoke-2026-08-31',
  modelHash: 'abc123',
  runtime: 'wasm',
  inputDimensions: [1024, 1024],
  processingTimeMs: 842,
  classStatistics: {
    undamaged: { pixelArea: 100, percentage: 100, estimatedRegions: 1 },
    minor: { pixelArea: 0, percentage: 0, estimatedRegions: 0 },
    major: { pixelArea: 0, percentage: 0, estimatedRegions: 0 },
    destroyed: { pixelArea: 0, percentage: 0, estimatedRegions: 0 },
  },
  regions: [
    { id: 'region-1', damageClass: 1, confidence: 0.87, normalizedBoundingBox: [0.1, 0.1, 0.2, 0.2], pixelArea: 100 },
  ],
  warnings: [],
};

const emptyResult: AssessmentResult = {
  ...baseResult,
  classStatistics: {
    undamaged: { pixelArea: 0, percentage: 0, estimatedRegions: 0 },
    minor: { pixelArea: 0, percentage: 0, estimatedRegions: 0 },
    major: { pixelArea: 0, percentage: 0, estimatedRegions: 0 },
    destroyed: { pixelArea: 0, percentage: 0, estimatedRegions: 0 },
  },
  regions: [],
};

describe('buildAssessmentJson', () => {
  it('produces JSON that validates against the schema', () => {
    const json = buildAssessmentJson(baseResult);
    const parsed = assessmentResultSchema.parse(JSON.parse(json));
    expect(parsed.modelVersion).toBe('siamese-smoke-2026-08-31');
  });

  it('exports an empty assessment without error', () => {
    const json = buildAssessmentJson(emptyResult);
    const parsed = JSON.parse(json);
    expect(parsed.regions).toEqual([]);
    expect(() => assessmentResultSchema.parse(parsed)).not.toThrow();
  });

  it('rejects a result that fails the schema', () => {
    const invalid = { ...baseResult, runtime: 'gpu' } as unknown as AssessmentResult;
    expect(() => buildAssessmentJson(invalid)).toThrow();
  });
});

describe('buildRegionsCsv', () => {
  it('writes one data row per region with a header row', () => {
    const csv = buildRegionsCsv(baseResult);
    const lines = csv.trim().split('\r\n');
    expect(lines).toHaveLength(2);
    expect(lines[0]).toBe('id,damageClass,confidence,boundingBoxX,boundingBoxY,boundingBoxWidth,boundingBoxHeight,pixelArea');
    expect(lines[1]).toBe('region-1,1,0.87,0.1,0.1,0.2,0.2,100');
  });

  it('produces only a header row for an empty assessment', () => {
    const csv = buildRegionsCsv(emptyResult);
    const lines = csv.trim().split('\r\n');
    expect(lines).toHaveLength(1);
    expect(lines[0]).toContain('id,damageClass');
  });

  it('escapes commas, quotes, and newlines in field values', () => {
    const withTrickyId: AssessmentResult = {
      ...baseResult,
      regions: [
        { id: 'weird,"id\nvalue', damageClass: 2, confidence: 0.5, normalizedBoundingBox: [0, 0, 1, 1], pixelArea: 10 },
      ],
    };
    const csv = buildRegionsCsv(withTrickyId);
    const dataLine = csv.trim().split('\r\n')[1];
    expect(dataLine.startsWith('"weird,""id\nvalue"')).toBe(true);
  });
});

describe('reportFilename', () => {
  it('includes the report kind, model version, timestamp, and extension', () => {
    const name = reportFilename('assessment', baseResult, 'json', new Date('2026-08-31T20:52:00.123Z'));
    expect(name).toBe('geoshield-assessment-siamese-smoke-2026-08-31-2026-08-31T20-52-00-123Z.json');
  });

  it('sanitizes unsafe characters out of the model version', () => {
    const name = reportFilename('regions', { ...baseResult, modelVersion: 'siamese v1/beta:2' }, 'csv', new Date('2026-01-01T00:00:00.000Z'));
    expect(name).toMatch(/^geoshield-regions-siamese_v1_beta_2-2026-01-01T00-00-00-000Z\.csv$/);
  });
});
