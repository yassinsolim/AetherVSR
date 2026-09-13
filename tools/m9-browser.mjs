import { spawn } from 'node:child_process';
import { mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';

export async function openNativeChrome(args = []) {
  process.env.PLAYWRIGHT_BROWSERS_PATH ??= resolve('.cache/m9/browsers');
  const { chromium } = await import('../.cache/m9/node_modules/playwright/index.mjs');
  const profile = mkdtempSync(join(tmpdir(), 'aethervsr-m9-'));
  const executable = process.env.M9_CHROME_EXECUTABLE_PATH ?? chromium.executablePath();
  const child = spawn(executable, ['--remote-debugging-port=0', `--user-data-dir=${profile}`,
    '--no-first-run', '--no-default-browser-check', '--disable-background-networking',
    '--disable-component-update', '--disable-sync', ...args, 'about:blank'], { stdio: ['ignore', 'ignore', 'pipe'] });
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
    browser = await chromium.connectOverCDP(endpoint, { noDefaults: true });
  } catch (error) {
    child.kill('SIGTERM');
    rmSync(profile, { recursive: true, force: true });
    throw error;
  }
  return { browser, context: browser.contexts()[0], executable,
    async close() {
      const timer = setTimeout(() => child.kill('SIGKILL'), 5000);
      try {
        await browser.close();
        if (child.exitCode === null) {
          const exited = new Promise(resolveExit => child.once('exit', resolveExit));
          child.kill('SIGTERM');
          await exited;
        }
      } finally {
        clearTimeout(timer);
        rmSync(profile, { recursive: true, force: true });
      }
    } };
}