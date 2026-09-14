import { afterAll, beforeAll, describe, expect, it } from 'vitest';
import { createHash } from 'node:crypto';
import { execFileSync } from 'node:child_process';
import { mkdtempSync, readFileSync, readdirSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { Script } from 'node:vm';
import ts from 'typescript';
import { MODEL_BYTES, MODEL_PATH, MODEL_SHA256 } from '../src/extension/protocol.js';

const root = fileURLToPath(new URL('../', import.meta.url));
const builder = new URL('../tools/build-extension.mjs', import.meta.url).href;
const sha256 = (bytes: Uint8Array | string) => createHash('sha256').update(bytes).digest('hex');
const artifactNames = ['content.js', 'manifest.json', MODEL_PATH, 'popup.css', 'popup.html', 'popup.js', 'service-worker.js'].sort();
interface Provenance {
  schemaVersion: number;
  generator: string;
  buildKind: string;
  sourceCommit: string;
  sourceDirty: boolean;
  manifestVersion: string;
  modelSha256: string;
  modelBytes: number;
  bundleSha256: string;
  totalBytes: number;
  files: Record<string, { sha256: string; bytes: number }>;
}
let temporary = '';
let production = '';
let repeated = '';
let testBuild = '';
const read = (directory: string, name: string) => readFileSync(join(directory, name));
const provenance = (directory: string) => JSON.parse(read(directory, 'build-provenance.json').toString()) as Provenance;

beforeAll(() => {
  temporary = mkdtempSync(join(tmpdir(), 'aethervsr-extension-test-'));
  production = join(temporary, 'production');
  repeated = join(temporary, 'repeated');
  testBuild = join(temporary, 'test');
  execFileSync(process.execPath, ['--input-type=module', '-e',
    `import {buildExtension} from ${JSON.stringify(builder)};
     await buildExtension({outdir:${JSON.stringify(production)}});
     await buildExtension({outdir:${JSON.stringify(repeated)}});
     await buildExtension({outdir:${JSON.stringify(testBuild)},test:true});`],
  { cwd: root, encoding: 'utf8', timeout: 60000 });
}, 60000);

afterAll(() => { if (temporary) rmSync(temporary, { recursive: true, force: true }); });

describe('built MV3 extension', () => {
  it('has exactly the preregistered permissions, CSP and packaged entrypoints', () => {
    const manifest: unknown = JSON.parse(read(production, 'manifest.json').toString());
    expect(manifest).toEqual({
      manifest_version: 3, name: 'AetherVSR', version: '0.1.0', minimum_chrome_version: '106',
      description: 'Local video enhancement for supported non-DRM web video.',
      permissions: ['activeTab', 'scripting', 'storage'],
      background: { service_worker: 'service-worker.js', type: 'module' },
      action: { default_popup: 'popup.html', default_title: 'AetherVSR' },
      content_security_policy: { extension_pages: "script-src 'self'; object-src 'none'" },
    });
    expect(read(production, 'manifest.json')).toEqual(readFileSync(join(root, 'src/extension/manifest.json')));
    expect(readdirSync(production).sort()).toEqual([
      'build-provenance.json', 'content.js', 'manifest.json', 'models', 'popup.css', 'popup.html', 'popup.js', 'service-worker.js',
    ]);
  });

  it('copies only the immutable production model without metadata rewriting', () => {
    const model = read(production, MODEL_PATH);
    expect(model.byteLength).toBe(MODEL_BYTES);
    expect(sha256(model)).toBe(MODEL_SHA256);
    expect(model).toEqual(readFileSync(join(root, 'public/models/aethersr-c16d2.json')));
    expect(readdirSync(join(production, 'models'))).toEqual(['production.json']);
    const worker = read(production, 'service-worker.js').toString();
    expect(worker).toContain(MODEL_PATH);
    expect(worker).toContain(MODEL_SHA256);
  });

  it('rejects modified production bytes before packaging', () => {
    const output = execFileSync(process.execPath, ['--input-type=module', '-e',
      `import {readFileSync} from 'node:fs'; import assert from 'node:assert/strict';
       import {verifyProductionModel} from ${JSON.stringify(builder)};
       const bytes=readFileSync('public/models/aethersr-c16d2.json');
       bytes[bytes.length-1]^=1;
       assert.throws(()=>verifyProductionModel(bytes),/integrity mismatch/);
       assert.throws(()=>verifyProductionModel(bytes.subarray(1)),/integrity mismatch/);
       console.log('rejected');`], { cwd: root, encoding: 'utf8' });
    expect(output.trim()).toBe('rejected');
  });

  it('produces byte-identical output on repeated builds', () => {
    for (const name of [...artifactNames, 'build-provenance.json']) expect(read(production, name)).toEqual(read(repeated, name));
  });

  it('pins source, manifest, model and all payload hashes without a provenance cycle', () => {
    const record = provenance(production);
    expect(record).toMatchObject({ schemaVersion: 1, generator: 'aethervsr-m10', buildKind: 'production',
      manifestVersion: '0.1.0', modelSha256: MODEL_SHA256, modelBytes: MODEL_BYTES });
    expect(record.sourceCommit).toMatch(/^[0-9a-f]{40,64}$/);
    expect(typeof record.sourceDirty).toBe('boolean');
    expect(Object.keys(record.files).sort()).toEqual(artifactNames);
    expect(record.files['build-provenance.json']).toBeUndefined();
    let total = 0;
    for (const name of artifactNames) {
      const bytes = read(production, name);
      expect(record.files[name]).toEqual({ sha256: sha256(bytes), bytes: bytes.byteLength });
      total += bytes.byteLength;
    }
    expect(record.totalBytes).toBe(total);
    expect(record.bundleSha256).toBe(sha256(Object.entries(record.files).map(([name, info]) => `${name}\0${info.sha256}\n`).join('')));
  });

  it('fully bundles the content IIFE and all local shader code with no debug or remote execution', () => {
    const content = read(production, 'content.js').toString();
    expect(() => new Script(content)).not.toThrow();
    expect(content).toContain('@compute');
    expect(content).toContain('texture_external');
    expect(content).not.toContain('Disable before test configuration');
    expect(content).not.toContain('__AETHERVSR_EXTENSION_TEST__');
    expect(content).not.toContain('diagnostic-processing-disabled');
    for (const name of ['content.js', 'service-worker.js', 'popup.js']) {
      const source = read(production, name).toString();
      for (const forbidden of ['sourceMappingURL', 'import.meta', '__AETHERVSR_TEST__', 'runtime-bench',
        '__aetherRuntime', 'unsafe-eval', 'debugger;', 'src/main.ts', '/@vite/client']) expect(source).not.toContain(forbidden);
      const parsed = ts.createSourceFile(name, source, ts.ScriptTarget.ESNext, true, ts.ScriptKind.JS);
      const inspect = (node: ts.Node): void => {
        expect(ts.isImportDeclaration(node)).toBe(false);
        if (ts.isExportDeclaration(node)) expect(node.moduleSpecifier).toBeUndefined();
        if (ts.isCallExpression(node)) {
          expect(node.expression.kind).not.toBe(ts.SyntaxKind.ImportKeyword);
          if (ts.isIdentifier(node.expression)) expect(['eval', 'require', 'Function']).not.toContain(node.expression.text);
        }
        if (ts.isNewExpression(node) && ts.isIdentifier(node.expression)) expect(node.expression.text).not.toBe('Function');
        ts.forEachChild(node, inspect);
      };
      inspect(parsed);
    }
  });

  it('labels the test artifact explicitly and keeps non-content payloads unchanged', () => {
    expect(provenance(testBuild).buildKind).toBe('test');
    expect(provenance(production).buildKind).toBe('production');
    for (const name of artifactNames.filter((name) => name !== 'content.js')) expect(read(testBuild, name)).toEqual(read(production, name));
    expect(read(testBuild, 'content.js')).not.toEqual(read(production, 'content.js'));
    expect(provenance(testBuild).bundleSha256).not.toBe(provenance(production).bundleSha256);
  });

  it('packages an accessible script-free HTML shell and explicit limitations', () => {
    const html = read(production, 'popup.html').toString();
    expect(html).toContain('<script type="module" src="popup.js"></script>');
    expect(html).not.toMatch(/\son\w+=|<script(?![^>]*src=)[^>]*>/i);
    expect(html).toContain('role="status"');
    expect(html).toContain('<legend>Mode for this origin</legend>');
    for (const text of ['Non-DRM', 'CORS', 'Iframes', 'closed shadow', 'Native video controls', 'fullscreen', 'PiP', 'WebGPU', 'hardware']) {
      expect(html).toContain(text);
    }
    const popup = read(production, 'popup.js').toString();
    expect(popup).not.toContain('innerHTML');
    expect(popup).not.toContain('setInterval');
    expect(popup).toContain('textContent');
    expect(popup).toContain('build-provenance.json');
  });
});