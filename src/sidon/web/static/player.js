// SPDX-License-Identifier: Apache-2.0
const players = new Set();
const clock = t => Number.isFinite(t) ? `${Math.floor(t / 60)}:${String(Math.floor(t % 60)).padStart(2, '0')}` : '0:00';

export class Player {
  constructor(container, title, {choose, download} = {}) {
    this.audio = new Audio(); this.audio.preload = 'metadata'; this.samples = null; this.url = null;
    this.root = document.createElement('article'); this.root.className = 'audio-editor';
    this.root.innerHTML = `<div class="audio-editor-head"><div class="audio-label"><i class="icon-file-audio" aria-hidden="true"></i><strong></strong><small></small></div><button class="icon-button download" aria-label="Download audio" hidden><i class="icon-download" aria-hidden="true"></i></button></div><div class="wave-stage"><canvas class="waveform" aria-hidden="true"></canvas><button class="wave-empty" type="button"><i class="icon-upload" aria-hidden="true"></i><strong>Drop your audio here</strong><span>WAV, FLAC, Opus, MP3, AAC/M4A, OGG</span></button></div><div class="audio-controls"><div class="audio-controls-side"><i class="icon-volume-2" aria-hidden="true"></i><input class="volume-slider" type="range" aria-label="Volume" min="0" max="1" step="0.05" value="1"></div><div class="transport"><button class="icon-button restart" aria-label="Restart" disabled><i class="icon-rotate-ccw" aria-hidden="true"></i></button><button class="play-button" aria-label="Play" disabled><i class="icon-play" aria-hidden="true"></i></button><button class="speed-button" aria-label="Playback speed" disabled>1×</button></div><div class="time-readout"><span class="current">0:00</span><span>/</span><span class="duration">0:00</span></div></div><input class="seek" type="range" min="0" max="1" step="0.001" value="0" aria-label="Seek audio" disabled>`;
    this.$ = s => this.root.querySelector(s);
    this.$('.audio-label strong').textContent = title;
    this.$('.wave-empty').hidden = !choose;
    if (choose) this.$('.wave-empty').onclick = choose;
    if (download) { this.$('.download').hidden = false; this.$('.download').onclick = download; }
    this.$('.play-button').onclick = async () => {
      if (this.audio.paused) {
        for (const player of players) if (player !== this) player.audio.pause();
        try { await this.audio.play(); } catch { this.$('.audio-label small').textContent = 'Playback unavailable in this browser. Download to listen.'; }
      } else this.audio.pause();
    };
    this.$('.restart').onclick = () => { this.audio.currentTime = 0; };
    this.$('.volume-slider').oninput = e => { this.audio.volume = +e.target.value; };
    this.$('.seek').oninput = e => { if (Number.isFinite(this.audio.duration)) this.audio.currentTime = +e.target.value * this.audio.duration; };
    this.$('.speed-button').onclick = e => { const rates = [1, 1.25, 1.5, 0.75]; this.audio.playbackRate = rates[(rates.indexOf(this.audio.playbackRate) + 1) % rates.length]; e.target.textContent = `${this.audio.playbackRate}×`; };
    this.audio.onloadedmetadata = () => { this.$('.duration').textContent = clock(this.audio.duration); };
    this.audio.ontimeupdate = () => { this.$('.current').textContent = clock(this.audio.currentTime); this.$('.seek').value = this.audio.currentTime / (this.audio.duration || 1); this.draw(); };
    this.audio.onplay = () => this.transport(true); this.audio.onpause = this.audio.onended = () => this.transport(false);
    this.observer = new ResizeObserver(() => this.draw()); this.observer.observe(this.root);
    container.append(this.root); players.add(this);
  }
  transport(playing) { this.$('.play-button i').className = playing ? 'icon-pause' : 'icon-play'; this.$('.play-button').ariaLabel = playing ? 'Pause' : 'Play'; }
  async load(blob, name = '') {
    this.audio.pause(); if (this.url) URL.revokeObjectURL(this.url);
    this.url = URL.createObjectURL(blob); this.audio.src = this.url; this.samples = null;
    this.audio.playbackRate = 1; this.$('.speed-button').textContent = '1×';
    this.$('.current').textContent = this.$('.duration').textContent = '0:00'; this.$('.seek').value = 0;
    this.$('.audio-label small').textContent = name; this.$('.wave-empty').hidden = true;
    for (const control of this.root.querySelectorAll('button,input')) control.disabled = false;
    this.draw();
    const url = this.url;
    let context;
    try {
      context = new AudioContext(); const buffer = await context.decodeAudioData(await blob.arrayBuffer());
      if (url !== this.url) return;
      const channel = buffer.getChannelData(0); const bins = 1600; const step = Math.max(1, Math.ceil(channel.length / bins));
      this.samples = new Float32Array(Math.ceil(channel.length / step));
      for (let i = 0; i < this.samples.length; i++) { let peak = 0; for (let j = i * step; j < Math.min(channel.length, (i + 1) * step); j++) peak = Math.max(peak, Math.abs(channel[j])); this.samples[i] = peak; }
      this.draw();
    } catch { /* Metadata and playback can still work when waveform decoding is unsupported. */ }
    finally { if (context) await context.close(); }
  }
  draw() {
    const canvas = this.$('canvas'), rect = canvas.getBoundingClientRect(); if (!rect.width) return;
    const ratio = window.devicePixelRatio || 1; canvas.width = rect.width * ratio; canvas.height = rect.height * ratio;
    const ctx = canvas.getContext('2d'); ctx.scale(ratio, ratio); const mid = rect.height / 2;
    ctx.strokeStyle = '#484b53'; ctx.beginPath(); ctx.moveTo(0, mid); ctx.lineTo(rect.width, mid); ctx.stroke();
    if (!this.samples) return;
    const played = this.audio.currentTime / (this.audio.duration || 1);
    for (let x = 0; x < rect.width; x += 3) { const i = Math.min(this.samples.length - 1, Math.floor(x / rect.width * this.samples.length)); const height = Math.max(1, Math.min(1, this.samples[i]) * (mid - 3)); ctx.fillStyle = x / rect.width < played ? '#ff7a1a' : '#b6b8c1'; ctx.fillRect(x, mid - height, 2, height * 2); }
  }
  dispose() { this.audio.pause(); this.audio.removeAttribute('src'); this.audio.load(); this.observer.disconnect(); if (this.url) URL.revokeObjectURL(this.url); this.url = null; players.delete(this); this.root.remove(); }
}
