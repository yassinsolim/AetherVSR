import { app, BrowserWindow, protocol, session } from 'electron';
import { readFile } from 'node:fs/promises';
import { join } from 'node:path';
import { allowedPermission, allowedRequest, APP_URL, CONTENT_SECURITY_POLICY, localAsset, SECURE_PREFERENCES } from './security.js';

declare const __DESKTOP_DIAGNOSTIC__: boolean;

protocol.registerSchemesAsPrivileged([{ scheme: 'aethervsr', privileges: {
  standard: true, secure: true, supportFetchAPI: true, corsEnabled: true, stream: true,
} }]);
app.setName('AetherVSR Desktop');
if (__DESKTOP_DIAGNOSTIC__ && process.env['AETHERVSR_TEST_PROFILE']) {
  app.setPath('userData', process.env['AETHERVSR_TEST_PROFILE']);
}

let window: BrowserWindow | null = null;
app.on('window-all-closed', () => app.quit());

void app.whenReady().then(async () => {
  const browsing = session.defaultSession;
  browsing.setPermissionCheckHandler((contents, permission, origin, details) =>
    allowedPermission(permission, contents?.getURL(), details.isMainFrame, origin));
  browsing.setPermissionRequestHandler((contents, permission, callback, details) =>
    callback(allowedPermission(permission, contents.getURL(), details.isMainFrame, details.requestingUrl)));
  browsing.on('will-download', (event, download) => { event.preventDefault(); download.cancel(); });
  browsing.webRequest.onBeforeRequest((details, callback) => callback({ cancel: !allowedRequest(details.url) }));
  browsing.protocol.handle('aethervsr', async request => {
    const asset = localAsset(request.url, request.method, request.headers.get('origin'));
    if (!asset) return new Response(null, { status: 403 });
    try {
      const bytes = await readFile(join(__dirname, asset.file));
      return new Response(request.method === 'HEAD' ? null : new Uint8Array(bytes), { headers: {
        'Content-Type': asset.type, 'Content-Security-Policy': CONTENT_SECURITY_POLICY,
        'X-Content-Type-Options': 'nosniff', 'Cache-Control': 'no-store',
      } });
    } catch { return new Response(null, { status: 404 }); }
  });
  window = new BrowserWindow({ width: 1100, height: 760, minWidth: 480, minHeight: 360,
    title: 'AetherVSR Desktop', backgroundColor: '#151916', show: false,
    webPreferences: { ...SECURE_PREFERENCES, devTools: __DESKTOP_DIAGNOSTIC__ } });
  window.webContents.setWindowOpenHandler(() => ({ action: 'deny' }));
  window.webContents.on('will-navigate', (event, url) => { if (url !== APP_URL) event.preventDefault(); });
  window.webContents.on('will-frame-navigate', event => { if (event.url !== APP_URL) event.preventDefault(); });
  window.webContents.on('will-redirect', event => event.preventDefault());
  window.webContents.on('will-attach-webview', event => event.preventDefault());
  window.webContents.on('render-process-gone', () => { window?.destroy(); app.quit(); });
  window.on('closed', () => { window = null; });
  await window.loadURL(APP_URL);
  window.show();
}).catch(error => { console.error(String(error)); app.exit(1); });