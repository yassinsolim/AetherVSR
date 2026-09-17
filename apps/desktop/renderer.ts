import { createElement, FolderOpen, Play, Pause, Volume2, VolumeX, Maximize, Minimize, Activity } from 'lucide';
import { DesktopSession, type DesktopSnapshot } from './session.js';
import type { RuntimeMode } from '../../src/core/upscale/runtime-controller.js';
import { DesktopDiagnostics } from '../../tools/m11/renderer-diagnostics.js';

declare const __DESKTOP_DIAGNOSTIC__: boolean;

const video = document.querySelector<HTMLVideoElement>('#source')!;
const canvas = document.querySelector<HTMLCanvasElement>('#output')!;
const file = document.querySelector<HTMLInputElement>('#file')!;
const open = document.querySelector<HTMLButtonElement>('#open')!;
const toggle = document.querySelector<HTMLButtonElement>('#play')!;
const mute = document.querySelector<HTMLButtonElement>('#mute')!;
const fullscreen = document.querySelector<HTMLButtonElement>('#fullscreen')!;
const seek = document.querySelector<HTMLInputElement>('#seek')!;
const volume = document.querySelector<HTMLInputElement>('#volume')!;
const mode = document.querySelector<HTMLSelectElement>('#mode')!;
const telemetry = document.querySelector<HTMLButtonElement>('#telemetry-toggle')!;
const details = document.querySelector<HTMLOutputElement>('#telemetry')!;
const status = document.querySelector<HTMLOutputElement>('#status')!;
const title = document.querySelector<HTMLElement>('#filename')!;
const clock = document.querySelector<HTMLOutputElement>('#clock')!;
const empty = document.querySelector<HTMLElement>('#empty')!;
const root = document.querySelector<HTMLElement>('#player')!;
let session: DesktopSession;
let showTelemetry = false;
let scrubbing = false;

function icon(button: HTMLButtonElement, symbol: typeof Play, label: string): void {
  const current = button.querySelector('svg'); current?.remove();
  button.prepend(createElement(symbol, { width: 18, height: 18, 'aria-hidden': 'true' }));
  button.setAttribute('aria-label', label); button.title = label;
}
function time(value: number | null): string {
  if (value === null || !Number.isFinite(value)) return '--:--';
  const whole = Math.floor(Math.max(0, value));
  return `${Math.floor(whole / 60)}:${String(whole % 60).padStart(2, '0')}`;
}
function render(snapshot: DesktopSnapshot): void {
  title.textContent = snapshot.name ?? 'AetherVSR Desktop';
  empty.hidden = snapshot.name !== null;
  toggle.disabled = snapshot.name === null;
  seek.disabled = snapshot.video.duration === null;
  if (!scrubbing) seek.value = String(snapshot.video.time);
  seek.max = String(snapshot.video.duration ?? 0);
  clock.value = `${time(snapshot.video.time)} / ${time(snapshot.video.duration)}`;
  volume.value = String(snapshot.video.volume);
  mode.value = snapshot.mode;
  if (toggle.dataset['paused'] !== String(snapshot.video.paused)) {
    icon(toggle, snapshot.video.paused ? Play : Pause, snapshot.video.paused ? 'Play' : 'Pause');
    toggle.dataset['paused'] = String(snapshot.video.paused);
  }
  const silent = snapshot.video.muted || snapshot.video.volume === 0;
  if (mute.dataset['muted'] !== String(silent)) { icon(mute, silent ? VolumeX : Volume2, silent ? 'Unmute' : 'Mute'); mute.dataset['muted'] = String(silent); }
  mute.setAttribute('aria-pressed', String(snapshot.video.muted));
  status.value = snapshot.error ? `Enhancement unavailable: ${snapshot.error}` : snapshot.name === null ? 'No file selected'
    : snapshot.pending ? 'Preparing enhancement' : snapshot.ready ? `${snapshot.runtime?.actualTier === 'neural' ? 'Neural 2x' : 'Baseline 2x'} | ${snapshot.video.width}x${snapshot.video.height}`
      : snapshot.video.paused ? 'Paused' : 'Original video';
  const runtime = session?.runtime;
  const stats = runtime?.pipeline.stats(performance.now());
  details.hidden = !showTelemetry;
  details.value = stats ? `${stats.sourceSize.width}x${stats.sourceSize.height} -> ${stats.targetSize.width}x${stats.targetSize.height} | rendered ${stats.meanRenderFps?.toFixed(1) ?? 'not measured'} FPS | GPU p95 ${stats.gpuPassMs?.p95.toFixed(2) ?? 'not measured'} ms`
    : 'GPU timing not measured';
}
const fail = (error: unknown) => { status.value = String(error); };
const diagnostics = __DESKTOP_DIAGNOSTIC__ ? new DesktopDiagnostics(video, canvas) : null;
session = new DesktopSession(video, canvas, { ...diagnostics?.options(), onChange: render });
diagnostics?.bind(() => session);
icon(open, FolderOpen, 'Open file'); icon(toggle, Play, 'Play'); icon(mute, Volume2, 'Mute');
icon(fullscreen, Maximize, 'Fullscreen'); icon(telemetry, Activity, 'Show telemetry');
open.addEventListener('click', () => file.click());
file.addEventListener('change', () => { const selected = file.files?.[0]; if (selected) { session.load(selected); render(session.snapshot()); } file.value = ''; });
toggle.addEventListener('click', () => { if (video.paused) void session.play().catch(fail); else session.pause(); });
mute.addEventListener('click', () => session.setMuted(!video.muted));
volume.addEventListener('input', () => session.setVolume(volume.valueAsNumber));
seek.addEventListener('pointerdown', () => { scrubbing = true; });
seek.addEventListener('change', () => { scrubbing = false; try { session.seek(seek.valueAsNumber); } catch (error) { fail(error); } });
seek.addEventListener('pointercancel', () => { scrubbing = false; });
mode.addEventListener('change', () => session.setMode(mode.value as RuntimeMode));
fullscreen.addEventListener('click', () => { void (document.fullscreenElement ? document.exitFullscreen() : root.requestFullscreen()).catch(fail); });
document.addEventListener('fullscreenchange', () => icon(fullscreen, document.fullscreenElement ? Minimize : Maximize, document.fullscreenElement ? 'Exit fullscreen' : 'Fullscreen'));
telemetry.addEventListener('click', () => { showTelemetry = !showTelemetry; telemetry.setAttribute('aria-pressed', String(showTelemetry)); render(session.snapshot()); });
video.addEventListener('timeupdate', () => render(session.snapshot()));
root.addEventListener('keydown', event => {
  if (event.code === 'Space' && event.target === root) { event.preventDefault(); toggle.click(); }
});
const interval = setInterval(() => { if (showTelemetry) render(session.snapshot()); }, 1000);
window.addEventListener('pagehide', () => { clearInterval(interval); session.destroy(); });
render(session.snapshot());

if (__DESKTOP_DIAGNOSTIC__) {
  Object.defineProperty(window, 'aethervsrRuntime', { get: () => ({ driver: session.runtime }) });
  Object.assign(window, { m11Desktop: { session: () => session, video, canvas, diagnostics,
    replace(options: { forceCopy?: boolean; observe?: boolean } = {}) {
      session.destroy(); session = new DesktopSession(video, canvas, { ...diagnostics?.options(options.observe),
        ...(options.forceCopy === undefined ? {} : { forceCopy: options.forceCopy }), onChange: render }); return session;
    } } });
}