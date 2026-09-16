import { spawn } from 'node:child_process';
import { mkdirSync, mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';

export async function openNativeChrome(args = [], { profileDirectory } = {}) {
  process.env.PLAYWRIGHT_BROWSERS_PATH ??= resolve('.cache/m9/browsers');
  const { chromium } = await import('../.cache/m9/node_modules/playwright/index.mjs');
  const profile = profileDirectory ? resolve(profileDirectory) : mkdtempSync(join(tmpdir(), 'aethervsr-m9-'));
  if (profileDirectory) mkdirSync(profile, { recursive: true });
  const executable = process.env.M9_CHROME_EXECUTABLE_PATH ?? chromium.executablePath();
  const child = spawn(executable, ['--remote-debugging-port=0', `--user-data-dir=${profile}`,
    '--no-first-run', '--no-default-browser-check', '--use-mock-keychain', '--disable-background-networking',
    '--disable-component-update', '--disable-sync', ...args, 'about:blank'], { stdio: ['ignore', 'ignore', 'pipe'] });
  const exited = new Promise(resolveExit => {
    child.once('exit', resolveExit);
    child.once('error', resolveExit);
  });
  const terminate = async () => {
    if (child.exitCode !== null || child.signalCode !== null) return;
    const timer = setTimeout(() => child.kill('SIGKILL'), 3000);
    child.kill('SIGTERM');
    try { await exited; } finally { clearTimeout(timer); }
  };
  let browser;
  try {
    const endpoint = await new Promise((resolveEndpoint, reject) => {
      let stderr = '';
      const timer = setTimeout(() => reject(new Error('Native Chrome endpoint timeout')), 15000);
      child.on('error', error => { clearTimeout(timer); reject(error); });
      child.stderr.on('data', chunk => {
        stderr = (stderr + chunk).slice(-8192);
        const match = stderr.match(/DevTools listening on (ws:\/\/\S+)/);
        if (match) { clearTimeout(timer); resolveEndpoint(match[1]); }
      });
      child.on('exit', code => { clearTimeout(timer); reject(new Error(`Native Chrome exited: ${code}`)); });
    });
    browser = await chromium.connectOverCDP(endpoint, { noDefaults: true, timeout: 15000 });
  } catch (error) {
    await terminate();
    if (!profileDirectory) rmSync(profile, { recursive: true, force: true });
    throw error;
  }
  return { browser, context: browser.contexts()[0], executable,
    async close() {
      let timer;
      try {
        await Promise.race([browser.close(), new Promise(resolveTimeout => { timer = setTimeout(resolveTimeout, 2000); })]);
      } finally {
        clearTimeout(timer);
        await terminate();
        if (!profileDirectory) rmSync(profile, { recursive: true, force: true });
      }
    } };
}