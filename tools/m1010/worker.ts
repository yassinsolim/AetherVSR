let sourceTab: number | null = null;
chrome.runtime.onMessage.addListener((message: unknown, sender, sendResponse) => {
  if (sender.id !== chrome.runtime.id || sender.url !== chrome.runtime.getURL('launcher.html')) return false;
  if (!message || typeof message !== 'object' || !('type' in message) || message.type !== 'research.open' ||
    !('surface' in message) || !['tab', 'window'].includes(String(message.surface))) return false;
  void (async () => {
    const selected = (await chrome.tabs.query({ active: true, currentWindow: true }))[0];
    sourceTab = selected?.id ?? null;
    const url = chrome.runtime.getURL('player.html');
    if (message.surface === 'window') return chrome.windows.create({ url, type: 'popup', width: 1000, height: 760 });
    return chrome.tabs.create({ url });
  })().then(result => sendResponse(result ? { ok: true, id: result.id } : { ok: false, error: 'Player surface was not created' }), error => sendResponse({ ok: false, error: String(error) }));
  return true;
});
chrome.runtime.onMessage.addListener((message: unknown, sender, sendResponse) => {
  if (sender.id !== chrome.runtime.id || !sender.url || sender.url.split('?')[0] !== chrome.runtime.getURL('player.html') ||
    !message || typeof message !== 'object' || !('type' in message) || message.type !== 'research.return') return false;
  if (sourceTab !== null) void chrome.tabs.update(sourceTab, { active: true }).then(() => sendResponse({ ok: true }), error => sendResponse({ ok: false, error: String(error) }));
  else sendResponse({ ok: false, error: 'No selected source tab' });
  return sourceTab !== null;
});