// SPDX-License-Identifier: Apache-2.0
const players = [...document.querySelectorAll('.audio-comparison audio')];
const icons = {
  play: '<path d="M7 4v16l13-8z"/>',
  pause: '<path d="M6 4h4v16H6zm8 0h4v16h-4z"/>',
  volume: '<path d="M3 9v6h4l5 4V5L7 9z"/><path d="M16 8a6 6 0 0 1 0 8m3-11a10 10 0 0 1 0 14" fill="none" stroke="currentColor" stroke-width="2"/>',
  mute: '<path d="M3 9v6h4l5 4V5L7 9z"/><path d="m16 9 6 6m0-6-6 6" fill="none" stroke="currentColor" stroke-width="2"/>',
};
const icon = name => `<svg viewBox="0 0 24 24" aria-hidden="true">${icons[name]}</svg>`;
const clock = seconds => Number.isFinite(seconds) ? `${Math.floor(seconds / 60)}:${String(Math.floor(seconds % 60)).padStart(2, '0')}` : '0:00';
for (const player of players) {
  const label = player.getAttribute('aria-label');
  const controls = document.createElement('div');
  controls.className = 'comparison-player';
  controls.setAttribute('role', 'group');
  controls.setAttribute('aria-label', `${label} controls`);
  controls.innerHTML = `<div class="player-transport"><button type="button" class="player-play">${icon('play')}</button><input class="player-seek" type="range" min="0" max="1" value="0" step="0.01" disabled><span class="player-clock">0:00 / 0:00</span></div><div class="player-options"><button type="button" class="player-speed">1×</button><div class="player-volume"><button type="button" class="player-mute">${icon('volume')}</button><input class="player-level" type="range" min="0" max="1" step="0.05" value="1"></div></div><p class="player-error" role="status" hidden></p>`;
  const find = selector => controls.querySelector(selector);
  const play = find('.player-play'), seek = find('.player-seek'), speed = find('.player-speed');
  const mute = find('.player-mute'), volume = find('.player-level'), error = find('.player-error');
  seek.setAttribute('aria-label', `Seek ${label}`);
  volume.setAttribute('aria-label', `Volume for ${label}`);
  speed.setAttribute('aria-label', `Playback speed for ${label}`);
  const sync = () => {
    play.innerHTML = icon(player.paused ? 'play' : 'pause');
    play.setAttribute('aria-label', `${player.paused ? 'Play' : 'Pause'} ${label}`);
    seek.disabled = !Number.isFinite(player.duration) || player.duration <= 0;
    seek.max = seek.disabled ? 1 : player.duration;
    seek.value = Number.isFinite(player.currentTime) ? player.currentTime : 0;
    seek.style.setProperty('--fill', `${seek.disabled ? 0 : Math.min(100, player.currentTime / player.duration * 100)}%`);
    seek.setAttribute('aria-valuetext', `${clock(player.currentTime)} of ${clock(player.duration)}`);
    find('.player-clock').textContent = `${clock(player.currentTime)} / ${clock(player.duration)}`;
    const silent = player.muted || player.volume === 0;
    mute.innerHTML = icon(silent ? 'mute' : 'volume');
    mute.setAttribute('aria-label', `${silent ? 'Unmute' : 'Mute'} ${label}`);
    mute.setAttribute('aria-pressed', String(silent));
    volume.value = player.muted ? 0 : player.volume;
    volume.style.setProperty('--fill', `${Number(volume.value) * 100}%`);
    speed.textContent = `${player.playbackRate}×`;
    speed.setAttribute('aria-label', `Playback speed for ${label}: ${player.playbackRate} times normal speed`);
  };
  play.addEventListener('click', async () => {
    error.hidden = true;
    if (!player.paused) { player.pause(); return; }
    try { await player.play(); }
    catch (failure) {
      if (failure.name !== 'AbortError') {
        error.textContent = 'Playback unavailable. Download the recording to listen.';
        error.hidden = false;
      }
    }
  });
  seek.addEventListener('input', () => { player.currentTime = Number(seek.value); sync(); });
  volume.addEventListener('input', () => { player.muted = false; player.volume = Number(volume.value); });
  mute.addEventListener('click', () => {
    if (player.volume === 0) { player.volume = 1; player.muted = false; }
    else player.muted = !player.muted;
  });
  speed.addEventListener('click', () => {
    const rates = [1, 1.25, 1.5, 2, .75];
    player.playbackRate = rates[(rates.indexOf(player.playbackRate) + 1) % rates.length];
  });
  for (const event of ['loadedmetadata', 'durationchange', 'timeupdate', 'play', 'pause', 'ended', 'volumechange', 'ratechange']) player.addEventListener(event, sync);
  player.addEventListener('play', () => {
    for (const other of players) if (other !== player) other.pause();
  });
  player.after(controls);
  player.controls = false;
  player.hidden = true;
  sync();
}
