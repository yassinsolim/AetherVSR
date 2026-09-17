import { describe, expect, it } from 'vitest';
import { allowedRequest, APP_ORIGIN, APP_URL, ASSETS, CONTENT_SECURITY_POLICY, localAsset, SECURE_PREFERENCES } from '../apps/desktop/security.js';

describe('desktop static asset and renderer boundaries', () => {
  it('allows only exact app assets with GET/HEAD and same-origin or headerless requests', () => {
    for (const path of Object.keys(ASSETS)) {
      expect(localAsset(`${APP_ORIGIN}${path}`)).not.toBeNull();
      expect(localAsset(`${APP_ORIGIN}${path}`, 'HEAD', APP_ORIGIN)).not.toBeNull();
    }
    for (const url of [APP_URL + '?file=/etc/passwd', APP_URL + '#remote', 'aethervsr://app/../index.html',
      'aethervsr://app/%69ndex.html', 'aethervsr://user@app/index.html', 'aethervsr://app:80/index.html',
      'file:///etc/passwd', 'https://example.com/', 'aethervsr://other/index.html']) expect(localAsset(url)).toBeNull();
    expect(localAsset(APP_URL, 'POST')).toBeNull();
    expect(localAsset(APP_URL, 'GET', 'https://example.com')).toBeNull();
    expect(localAsset(APP_URL, 'GET', 'null')).toBeNull();
  });

  it('allows app-owned blob resources without enabling arbitrary files or remote documents', () => {
    expect(allowedRequest('blob:aethervsr://app/01234567-0123-4123-8123-0123456789ab')).toBe(true);
    for (const url of ['blob:https://example.com/id', 'blob:null/id', 'file:///video.mp4', 'https://example.com/video.mp4',
      'aethervsr://app/unknown.js', 'data:text/html,hello']) expect(allowedRequest(url)).toBe(false);
  });

  it('requires sandboxed isolated rendering and restrictive local content', () => {
    expect(SECURE_PREFERENCES).toEqual({ nodeIntegration: false, contextIsolation: true, sandbox: true,
      webSecurity: true, webviewTag: false, allowRunningInsecureContent: false, experimentalFeatures: false });
    for (const directive of ["default-src 'none'", "script-src 'self'", "media-src blob:", "frame-src 'none'", "worker-src 'none'", "base-uri 'none'", "form-action 'none'"]) expect(CONTENT_SECURITY_POLICY).toContain(directive);
    expect(CONTENT_SECURITY_POLICY).not.toMatch(/unsafe|https:/);
  });
});