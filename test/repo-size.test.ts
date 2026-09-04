import { describe, expect, it } from 'vitest';
import { execFileSync } from 'node:child_process';

/**
 * Guards against re-committing large generated media.
 *
 * Milestone 4.5 committed 397 MB of regenerable evaluation frames and video,
 * which took a history rewrite and a force-push to undo. Nothing in the process
 * objected at the time: the files were legitimate outputs, they were needed by
 * the benchmarks, and each individual commit looked reasonable. What was missing
 * was anything that noticed the aggregate.
 *
 * This runs in CI on every push, against the committed tree only.
 */

const MAX_FILE_BYTES = 8 * 1024 * 1024;
const MAX_TRACKED_BYTES = 40 * 1024 * 1024;

/**
 * Deliberate exceptions, each with a reason. Anything not listed here and over
 * the limit fails, so adding large media is a decision someone has to write
 * down rather than something that happens by accident.
 */
const ALLOWED_LARGE = new Set<string>([
  // The browser harness plays these; a clean checkout must be able to run the
  // production pipeline without a Python step or a network fetch.
  'public/media/aethervsr-testclip-720p30-h264.mp4',
  'public/media/aethervsr-testclip-720p30-vp9.webm',
  'public/media/aethervsr-testclip-720p60-h264.mp4',
  'public/media/aethervsr-testclip-720p60-vp9.webm',
]);

interface Tracked {
  readonly path: string;
  readonly bytes: number;
}

function trackedFiles(): Tracked[] {
  // -s gives the object size directly, so this never reads the worktree and
  // never depends on files a contributor happens to have generated locally.
  const out = execFileSync('git', ['ls-tree', '-r', '-l', '--full-name', 'HEAD'], {
    encoding: 'utf8',
    maxBuffer: 64 * 1024 * 1024,
  });
  const files: Tracked[] = [];
  for (const line of out.split('\n')) {
    if (!line) continue;
    // <mode> <type> <object> <size>\t<path>
    const tab = line.indexOf('\t');
    if (tab < 0) continue;
    const size = line.slice(0, tab).trim().split(/\s+/)[3];
    if (size === undefined || size === '-') continue;
    files.push({ path: line.slice(tab + 1), bytes: Number(size) });
  }
  return files;
}

describe('repository size discipline', () => {
  const files = trackedFiles();

  it('reads the committed tree', () => {
    expect(files.length).toBeGreaterThan(50);
  });

  it('commits no unapproved file over 8 MB', () => {
    const offenders = files
      .filter((f) => f.bytes > MAX_FILE_BYTES && !ALLOWED_LARGE.has(f.path))
      .map((f) => `${f.path} (${(f.bytes / 1e6).toFixed(1)} MB)`);
    // Naming them makes the failure actionable rather than just red.
    expect(offenders).toEqual([]);
  });

  it('keeps the whole tracked tree under 40 MB', () => {
    const total = files.reduce((n, f) => n + f.bytes, 0);
    const worst = [...files].sort((a, b) => b.bytes - a.bytes).slice(0, 5);
    const detail = worst.map((f) => `${f.path}=${(f.bytes / 1e6).toFixed(1)}MB`).join(' ');
    expect(
      total,
      `tracked tree is ${(total / 1e6).toFixed(1)} MB; largest: ${detail}. ` +
        'Large benchmark media belongs behind a manifest and a generator, not in git.',
    ).toBeLessThan(MAX_TRACKED_BYTES);
  });

  it('does not track the regenerable corpora', () => {
    // These have deterministic producers and hash-pinned manifests, so a clone
    // rebuilds them. Re-adding the pixels is the exact 4.5 mistake.
    const regenerable = files.filter(
      (f) =>
        /^data\/video\/.+\.(png|mp4|webm)$/.test(f.path) ||
        /^data\/eval-independent\/.+\.(jpe?g|png)$/.test(f.path) ||
        /^data\/corpus\/.+\.(jpe?g|png)$/.test(f.path) ||
        /^data\/captured\/.+\.(png|mp4|webm|mkv|mov|y4m)$/.test(f.path),
    );
    expect(regenerable.map((f) => f.path)).toEqual([]);
  });
});
