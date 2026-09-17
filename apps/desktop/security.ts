export const APP_ORIGIN = 'aethervsr://app';
export const APP_URL = `${APP_ORIGIN}/index.html`;
export const CONTENT_SECURITY_POLICY = "default-src 'none'; script-src 'self'; style-src 'self'; img-src 'self'; media-src blob:; connect-src 'self' blob:; font-src 'none'; object-src 'none'; frame-src 'none'; worker-src 'none'; base-uri 'none'; form-action 'none'";
export const ASSETS: Readonly<Record<string, { file: string; type: string }>> = {
  '/index.html': { file: 'index.html', type: 'text/html; charset=utf-8' },
  '/renderer.js': { file: 'renderer.js', type: 'text/javascript; charset=utf-8' },
  '/player.css': { file: 'player.css', type: 'text/css; charset=utf-8' },
  '/models/production.json': { file: 'models/production.json', type: 'application/json' },
};

export function localAsset(url: string, method = 'GET', origin: string | null = null) {
  if (method !== 'GET' && method !== 'HEAD') return null;
  if (origin !== null && origin !== APP_ORIGIN) return null;
  const asset = Object.entries(ASSETS).find(([path]) => url === `${APP_ORIGIN}${path}`);
  return asset?.[1] ?? null;
}

export function allowedRequest(url: string): boolean {
  if (localAsset(url)) return true;
  return /^blob:aethervsr:\/\/app\/[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/.test(url);
}

export function allowedPermission(permission: string, documentUrl: string | undefined, isMainFrame: boolean, requester: string): boolean {
  return permission === 'fullscreen' && documentUrl === APP_URL && isMainFrame &&
    (requester === APP_ORIGIN || requester === APP_URL);
}

export const SECURE_PREFERENCES = {
  nodeIntegration: false, contextIsolation: true, sandbox: true,
  webSecurity: true, webviewTag: false, allowRunningInsecureContent: false,
  experimentalFeatures: false,
} as const;