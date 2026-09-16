document.getElementById('acquire')!.addEventListener('click', () => {
  void chrome.runtime.sendMessage({ type: 'research.acquire' }).then((result: unknown) => { document.getElementById('status')!.textContent = JSON.stringify(result); });
});

for (const type of ['tab', 'window']) document.getElementById(type)!.addEventListener('click', () => {
  void chrome.runtime.sendMessage({ type: 'research.open', surface: type }).then((result: unknown) => {
    document.getElementById('status')!.textContent = JSON.stringify(result);
  });
});