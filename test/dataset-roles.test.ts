import { describe, expect, it } from 'vitest';
import { existsSync, readFileSync } from 'node:fs';

/**
 * Enforces the dataset role boundaries described in docs/DATASET-ROLES.md.
 *
 * The failure this prevents is silent. An evaluation clip that drifts into a
 * training corpus does not raise an error, it just returns a better number, and
 * every downstream claim inherits the contamination without anything looking
 * wrong. Milestone 5.5 found a real instance of the weaker version of this: two
 * clips from one shoot split across train and validation, which no file-level
 * check could see.
 *
 * So the checks here are on identity, not on filenames — a re-upload under a new
 * name is the same footage.
 */

type Clip = {
  id: string;
  title?: string;
  creator?: string;
  shootId?: string;
  source_url?: string;
  source_sha256?: string;
  description_url?: string;
  licence?: string;
};
type Manifest = { role?: string; schema?: string; clips?: Clip[] };

const TRAINING = ['../data/captured-train/manifest.json', '../data/captured-train-v2/manifest.json'];
const EVALUATION = [
  '../data/captured/manifest.json',
  '../data/captured-faces/manifest.json',
  '../data/captured-val/manifest.json',
  '../data/captured-confirm/manifest.json',
];

function load(rel: string): Manifest | null {
  const url = new URL(rel, import.meta.url);
  if (!existsSync(url)) return null;
  return JSON.parse(readFileSync(url, 'utf8')) as Manifest;
}

function clipsOf(paths: string[]): { path: string; clip: Clip }[] {
  const out: { path: string; clip: Clip }[] = [];
  for (const p of paths) {
    const m = load(p);
    if (m === null) continue;
    for (const clip of m.clips ?? []) out.push({ path: p, clip });
  }
  return out;
}

/** Mirrors tools/corpus_schema.py:normalise_creator exactly. Commons credits are
 *  free text; reduce to the first credited party, since that is the operator
 *  whose camera and encoder the corpus inherits. Kept in sync deliberately —
 *  the two implementations drifted once and the divergence hid a real overlap. */
function normaliseCreator(name: string | undefined): string {
  let s = (name ?? '').normalize('NFKC').toLowerCase();
  // `split` is typed as possibly-undefined under noUncheckedIndexedAccess;
  // both splits always yield at least one element.
  s = s.split('\n')[0] ?? '';
  s = s.split(/(?:audio|music|sound|editing|edit|schild|text)\s*[:-]/)[0] ?? '';
  s = s.replace(/^\s*(?:video|foto|photo|footage|film|by|author|creator|image)\s*[:-]\s*/, ' ');
  s = s.replace(/\(.*?\)/g, ' ');
  s = s.replace(/[^\p{L}\p{N}\s]/gu, ' ');
  s = s.split(/\s+/).filter(Boolean).join(' ');
  return s || 'unattributed';
}

describe('dataset roles', () => {
  it.each([...TRAINING, ...EVALUATION])('%s declares a role', (rel) => {
    const m = load(rel);
    if (m === null) return; // corpus not present in this checkout
    expect(m.role).toBeDefined();
  });

  it('no training clip shares a content hash with any evaluation clip', () => {
    const evalHashes = new Set(
      clipsOf(EVALUATION)
        .map(({ clip }) => clip.source_sha256)
        .filter((h): h is string => Boolean(h)),
    );
    const collisions = clipsOf(TRAINING)
      .filter(({ clip }) => clip.source_sha256 && evalHashes.has(clip.source_sha256))
      .map(({ path, clip }) => `${path}:${clip.id}`);
    expect(collisions).toEqual([]);
  });

  it('no training clip shares a source URL or media page with any evaluation clip', () => {
    const evalRefs = new Set<string>();
    for (const { clip } of clipsOf(EVALUATION)) {
      if (clip.source_url) evalRefs.add(clip.source_url);
      if (clip.description_url) evalRefs.add(clip.description_url);
      if (clip.title) evalRefs.add(clip.title);
    }
    const collisions = clipsOf(TRAINING)
      .filter(
        ({ clip }) =>
          (clip.source_url && evalRefs.has(clip.source_url)) ||
          (clip.description_url && evalRefs.has(clip.description_url)) ||
          (clip.title && evalRefs.has(clip.title)),
      )
      .map(({ path, clip }) => `${path}:${clip.id}`);
    expect(collisions).toEqual([]);
  });

  it('no shoot spans training and evaluation', () => {
    const evalShoots = new Set(
      clipsOf(EVALUATION)
        .map(({ clip }) => clip.shootId)
        .filter((s): s is string => Boolean(s)),
    );
    const collisions = clipsOf(TRAINING)
      .filter(({ clip }) => clip.shootId && evalShoots.has(clip.shootId))
      .map(({ path, clip }) => `${path}:${clip.id} (${clip.shootId})`);
    expect(collisions).toEqual([]);
  });

  it('reports creator overlap between training and every evaluation corpus', () => {
    // Not an assertion that overlap is zero - it is not, and pretending
    // otherwise would be worse than recording it. Milestone 6 shipped with
    // training sharing creators with the regression and faces benchmarks
    // (MrAyrit shot both a training clip and a benchmark clip on the same
    // river with the same drone a month apart). The confirmation set, which
    // carries the headline, is disjoint and asserted separately below.
    //
    // This test exists so the overlap cannot grow silently: it pins the known
    // set, and any new collision fails until it is examined and recorded.
    const KNOWN = {
      '../data/captured/manifest.json': ['mrayrit', 'capricorn4049', 'pantheraleo1359531'],
      '../data/captured-faces/manifest.json': ['nasa johnson space center', 'nasa kennedy space center'],
    } as Record<string, string[]>;
    const trainCreators = new Set(
      clipsOf(TRAINING).map(({ clip }) => normaliseCreator(clip.creator)),
    );
    for (const [path, known] of Object.entries(KNOWN)) {
      const m = load(path);
      if (m === null) continue;
      const shared = [
        ...new Set(
          (m.clips ?? [])
            .map((c) => normaliseCreator(c.creator))
            .filter((c) => c !== 'unattributed' && trainCreators.has(c)),
        ),
      ].sort();
      expect(shared).toEqual([...known].sort());
    }
  });

  it('the confirmation set shares no creator with training', () => {
    // Stricter than the others by design: the confirmation set exists to answer
    // whether the model generalises to footage from people it never learned
    // from, which a shared creator would quietly undermine.
    const confirm = load('../data/captured-confirm/manifest.json');
    if (confirm === null) return;
    const trainCreators = new Set(
      clipsOf(TRAINING).map(({ clip }) => normaliseCreator(clip.creator)),
    );
    const collisions = (confirm.clips ?? [])
      .filter((c) => trainCreators.has(normaliseCreator(c.creator)))
      .map((c) => `${c.id} (${c.creator})`);
    expect(collisions).toEqual([]);
  });

  it('every clip in every corpus carries a shoot id', () => {
    const missing = clipsOf([...TRAINING, ...EVALUATION])
      .filter(({ clip }) => !clip.shootId)
      .map(({ path, clip }) => `${path}:${clip.id}`);
    // v1 predates the schema and is retained only for the scaling curve.
    const relevant = missing.filter((m) => !m.includes('captured-train/manifest'));
    expect(relevant).toEqual([]);
  });
});
