import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';

/**
 * The split that Milestone 4 shipped pooled several patches from every
 * photograph and shuffled the pool, so the same source image landed in both
 * training and validation. Replaying it on the 495-image corpus:
 *
 *   images contributing to both train and val   231  (46.7%)
 *   val patches whose source also appears in train   297/297  (100.00%)
 *
 * Every validation patch came from a photograph the model had trained on.
 * These tests exist so that cannot silently return. They run in CI, against
 * the committed split manifest, with no Python and no GPU.
 */

interface SplitEntry {
  readonly file: string;
  readonly sha256: string;
  readonly source_url: string | null;
  readonly licence: string | null;
}

interface SplitManifest {
  readonly schema: string;
  readonly unit: string;
  readonly salt: string;
  readonly counts: Record<string, number>;
  readonly clusters: number;
  readonly phashThreshold: number;
  readonly total: number;
  readonly splits: Record<'train' | 'val' | 'test', SplitEntry[]>;
}

const split = JSON.parse(
  readFileSync(new URL('../data/splits/corpus-v2.json', import.meta.url), 'utf8'),
) as SplitManifest;

const NAMES = ['train', 'val', 'test'] as const;
const PAIRS: readonly (readonly ['train' | 'val' | 'test', 'train' | 'val' | 'test'])[] = [
  ['train', 'val'],
  ['train', 'test'],
  ['val', 'test'],
];

const hashes = (name: (typeof NAMES)[number]): Set<string> =>
  new Set(split.splits[name].map((e) => e.sha256));
const files = (name: (typeof NAMES)[number]): Set<string> =>
  new Set(split.splits[name].map((e) => e.file));

function intersect(a: Set<string>, b: Set<string>): string[] {
  return [...a].filter((v) => b.has(v));
}

describe('source-level dataset split', () => {
  it('is the expected schema and splits by source identity, not by patch', () => {
    expect(split.schema).toBe('aethervsr.split/1');
    expect(split.salt.length).toBeGreaterThan(0);
  });

  /**
   * Hash disjointness is not enough. Splitting per image left 36 near-duplicate
   * pairs straddling splits - the same manuscript folio scanned twice, the same
   * demonstration photographed seconds apart, several at dHash distance 2-5
   * with identical Commons titles. Assignment is by perceptual cluster now, and
   * a corpus whose clusters equal its image count means clustering silently
   * stopped working.
   */
  it('assigns perceptual clusters rather than individual images', () => {
    expect(split.unit).toContain('cluster');
    expect(split.clusters).toBeGreaterThan(0);
    expect(split.clusters).toBeLessThan(split.total);
    expect(split.phashThreshold).toBeGreaterThan(0);
  });

  it.each(PAIRS)('%s and %s share no source content hash', (a, b) => {
    // The load-bearing assertion. Content hash is the identity that matters:
    // a photograph re-encoded under a new filename is still the same picture.
    expect(intersect(hashes(a), hashes(b))).toEqual([]);
  });

  it.each(PAIRS)('%s and %s share no filename', (a, b) => {
    expect(intersect(files(a), files(b))).toEqual([]);
  });

  it('contains no duplicate source within a single split', () => {
    for (const name of NAMES) {
      expect(hashes(name).size).toBe(split.splits[name].length);
      expect(files(name).size).toBe(split.splits[name].length);
    }
  });

  it('accounts for every source exactly once', () => {
    const counted = NAMES.reduce((n, name) => n + split.splits[name].length, 0);
    expect(counted).toBe(split.total);
    const all = new Set(NAMES.flatMap((name) => [...hashes(name)]));
    expect(all.size).toBe(split.total);
  });

  it('declares counts that match its contents', () => {
    for (const name of NAMES) {
      expect(split.counts[name]).toBe(split.splits[name].length);
    }
  });

  it('leaves every split large enough to mean something', () => {
    // A validation split of five images makes checkpoint selection noise.
    for (const name of NAMES) {
      expect(split.splits[name].length).toBeGreaterThan(30);
    }
  });

  it('records a source identity for every entry', () => {
    for (const name of NAMES) {
      for (const entry of split.splits[name]) {
        expect(entry.sha256).toMatch(/^[0-9a-f]{64}$/);
        expect(entry.file.length).toBeGreaterThan(0);
      }
    }
  });
});
