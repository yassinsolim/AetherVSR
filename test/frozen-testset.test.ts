import { describe, expect, it } from 'vitest';
import { readFileSync, existsSync } from 'node:fs';

/**
 * Enforces the Milestone 5 test-set freeze in CI.
 *
 * The captured corpus is the only evidence the project has about real web
 * video. If a clip from it ever enters a training corpus the headline becomes
 * unfalsifiable, and nothing else in the repository would notice: the training
 * manifests and the test manifest are separate files that no other check
 * compares.
 *
 * This runs against committed manifests only — no media, no Python, no GPU.
 */

interface Frozen {
  readonly clipIds: readonly string[];
  readonly sourceSha256: Record<string, string>;
  readonly testsetDigest: string;
  readonly clipCount: number;
}

interface TrainingManifest {
  readonly clips?: readonly { readonly id: string; readonly source_sha256?: string }[];
  readonly images?: readonly { readonly sha256?: string }[];
}

const frozen = JSON.parse(
  readFileSync(new URL('../data/captured/FROZEN.json', import.meta.url), 'utf8'),
) as Frozen;

/** Every corpus that may legitimately be trained on. */
const TRAINING_MANIFESTS = [
  '../data/corpus/manifest.json',
  '../data/captured-train/manifest.json',
];

function loadIfPresent(rel: string): TrainingManifest | null {
  const url = new URL(rel, import.meta.url);
  return existsSync(url) ? (JSON.parse(readFileSync(url, 'utf8')) as TrainingManifest) : null;
}

describe('Milestone 5 captured test set stays frozen', () => {
  it('declares the ten clips it froze', () => {
    expect(frozen.clipCount).toBe(10);
    expect(frozen.clipIds).toHaveLength(10);
    expect(Object.keys(frozen.sourceSha256)).toHaveLength(10);
  });

  it.each(TRAINING_MANIFESTS)('%s shares no source hash with the test set', (rel) => {
    const manifest = loadIfPresent(rel);
    if (manifest === null) return; // corpus not present in this checkout
    const testHashes = new Set(Object.values(frozen.sourceSha256));
    const trainHashes = [
      ...(manifest.clips ?? []).map((c) => c.source_sha256),
      ...(manifest.images ?? []).map((i) => i.sha256),
    ].filter((h): h is string => typeof h === 'string');
    // Content hash is the identity that matters: a re-uploaded copy under a new
    // filename is still the same footage.
    expect(trainHashes.filter((h) => testHashes.has(h))).toEqual([]);
  });

  it.each(TRAINING_MANIFESTS)('%s shares no clip id with the test set', (rel) => {
    const manifest = loadIfPresent(rel);
    if (manifest === null) return;
    const testIds = new Set(frozen.clipIds);
    expect((manifest.clips ?? []).map((c) => c.id).filter((id) => testIds.has(id))).toEqual([]);
  });

  it('the live captured manifest still matches the freeze', () => {
    const live = JSON.parse(
      readFileSync(new URL('../data/captured/manifest.json', import.meta.url), 'utf8'),
    ) as { clips: { id: string; source_sha256: string }[] };
    // Catches a clip being added or removed after the freeze, which is how a
    // corpus quietly becomes tuned to its own results.
    expect(live.clips.map((c) => c.id).sort()).toEqual([...frozen.clipIds].sort());
    for (const clip of live.clips) {
      expect(clip.source_sha256).toBe(frozen.sourceSha256[clip.id]);
    }
  });
});
