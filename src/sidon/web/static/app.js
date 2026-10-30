// SPDX-License-Identifier: Apache-2.0
import {Player} from './player.js';
const $ = s => document.querySelector(s);
let apiKey = '', view = 'enhance', file = null, activeJob = null, running = false, gpuBusy = false, lastStatus = null;
let outputPlayers = [], pollTimer, gpuTimer, pollingStarted = 0, pollGeneration = 0;
let authenticationRequired = false;
const authenticationError = () => Object.assign(new Error('An API key is required by this server. Enter it in System and click Connect.'), {status: 401});
const message = (text, error = false) => { for (const selector of ['#job-status', '#system-status']) { $(selector).textContent = text; $(selector).dataset.tone = error ? 'error' : ''; } };
async function request(url, options = {}) {
  if (authenticationRequired) throw authenticationError();
  const response = await fetch(url, {...options, headers: {...options.headers, ...(apiKey ? {Authorization: `Bearer ${apiKey}`} : {})}});
  if (!response.ok) {
    let detail; try { detail = (await response.json()).detail; } catch {}
    if (response.status === 401) {
      authenticationRequired = true; clearTimeout(gpuTimer);
      $('#runtime-state').textContent = 'Authentication required';
      $('#runtime-detail').textContent = 'Enter the server API key in System';
      $('#runtime-badge').dataset.state = 'error';
      throw authenticationError();
    }
    throw Object.assign(new Error(typeof detail === 'string' ? detail : `Request failed (${response.status}). Check the file and settings.`), {status: response.status});
  }
  return response;
}
const json = async (url, options) => (await request(url, options)).json();
const jsonOptions = body => ({headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body)});
function guarded(fn) { return async (...args) => { try { await fn(...args); } catch (error) { message(error.message, true); } }; }
async function download(url, filename) {
  const blob = await (await request(url)).blob(); const object = URL.createObjectURL(blob);
  const anchor = document.createElement('a'); anchor.href = object; anchor.download = filename; anchor.click(); setTimeout(() => URL.revokeObjectURL(object), 10000);
}
const inputPlayer = new Player($('#input-player'), 'Original audio', {choose: () => $('#file').click(), download: guarded(() => activeJob ? download(`/api/jobs/${activeJob}/files/input.wav`, 'original.wav') : localDownload())});
function localDownload() { if (!file) return; const url = URL.createObjectURL(file), a = document.createElement('a'); a.href = url; a.download = file.name; a.click(); setTimeout(() => URL.revokeObjectURL(url), 10000); }
async function selectFile(selected) {
  if (!selected || running) return;
  if (lastStatus && selected.size > lastStatus.limits.upload_mb * 1024 ** 2) throw new Error(`Choose a file smaller than ${lastStatus.limits.upload_mb} MB.`);
  file = selected; beginJob();
  await inputPlayer.load(file, file.name);
  if (file !== selected) return;
  $('#process').disabled = false; message('Ready to process.');
}
$('#choose').onclick = () => $('#file').click(); $('#file').onchange = guarded(e => selectFile(e.target.files[0]));
inputPlayer.root.ondragover = e => { e.preventDefault(); inputPlayer.root.classList.add('dragging'); };
inputPlayer.root.ondragleave = () => inputPlayer.root.classList.remove('dragging');
inputPlayer.root.ondrop = guarded(async e => { e.preventDefault(); inputPlayer.root.classList.remove('dragging'); await selectFile(e.dataTransfer.files[0]); });
function modeChanged() {
  const mode = $('#mode').value, conversion = mode === 'convert', separation = ['separate','separate_enhance'].includes(mode);
  const lossless = ['wav','flac'].includes($('#output-format').value);
  $('#strength-field').hidden = !['enhance','separate_enhance'].includes(mode);
  $('#mode-help').textContent = conversion ? 'Change audio format without loading a model or applying enhancement.' : separation ? 'Separate a two-person mixture with MossFormer2. Add enhancement to restore each voice afterwards.' : 'Restore noisy or degraded speech with Sidon. Best suited to a single voice.';
  $('#output-help').textContent = conversion ? (lossless ? 'Keeps the input sample rate and channels.' : 'Compressed output at 48 kHz. MP3 supports up to two channels.') : mode === 'separate' ? `Two mono tracks at ${lossless ? 16 : 48} kHz.` : separation ? 'Two enhanced mono tracks at 48 kHz.' : 'Enhanced audio: mono, 48 kHz.';
  $('#process span').textContent = conversion ? 'Convert audio' : mode === 'enhance' ? 'Enhance audio' : mode === 'separate' ? 'Split speakers' : 'Split & enhance';
  $('#workflow-title').textContent = conversion ? 'Your audio, in the format you need' : separation ? 'Give each speaker their own track' : 'Give your voice a cleaner finish';
}
$('#mode').onchange = () => { changeView(['enhance','convert'].includes($('#mode').value) ? $('#mode').value : 'separate', false); };
$('#output-format').onchange = modeChanged;
$('#strength').oninput = () => { $('#strength-value').textContent = `${Math.round(+$('#strength').value * 100)}%`; };
function changeView(next, setMode = true) {
  if (!['enhance','separate','convert','api','system'].includes(next)) next = 'enhance'; view = next;
  sessionStorage.setItem('sidon-tab', view); history.replaceState(null, '', `#${view}`);
  $('#view-process').hidden = !['enhance','separate','convert'].includes(view); $('#view-api').hidden = view !== 'api'; $('#view-system').hidden = view !== 'system';
  $('#view-process').setAttribute('aria-labelledby', `tab-${['separate','convert'].includes(view) ? view : 'enhance'}`);
  for (const tab of document.querySelectorAll('[role=tab]')) { const active = tab.dataset.view === view; tab.classList.toggle('active', active); tab.ariaSelected = String(active); tab.tabIndex = active ? 0 : -1; }
  if (setMode && view === 'enhance') $('#mode').value = 'enhance';
  if (setMode && view === 'convert') $('#mode').value = 'convert';
  if (setMode && view === 'separate' && ['enhance','convert'].includes($('#mode').value)) $('#mode').value = 'separate';
  modeChanged(); scheduleGpu(); if (view === 'system') refreshHistory().catch(() => {});
}
document.querySelectorAll('[data-view]').forEach(el => { el.onclick = e => { e.preventDefault(); changeView(el.dataset.view); }; });
$('.app-tabs').onkeydown = e => { const tabs = [...document.querySelectorAll('[role=tab]')]; let position = tabs.indexOf(document.activeElement); if (position < 0) return; if (['ArrowLeft','ArrowRight','Home','End'].includes(e.key)) { e.preventDefault(); position = e.key === 'Home' ? 0 : e.key === 'End' ? tabs.length - 1 : (position + (e.key === 'ArrowRight' ? 1 : -1) + tabs.length) % tabs.length; tabs[position].click(); tabs[position].focus(); } };
function collapsed(value) { document.documentElement.dataset.headerCollapsed = String(value); sessionStorage.setItem('sidon-header', String(value)); $('#hero-toggle').ariaExpanded = String(!value); $('#hero-toggle').ariaLabel = value ? 'Expand header' : 'Collapse header'; $('#hero-toggle i').className = value ? 'icon-chevron-down' : 'icon-chevron-up'; }
$('#hero-toggle').onclick = () => collapsed(document.documentElement.dataset.headerCollapsed !== 'true'); collapsed(sessionStorage.getItem('sidon-header') === 'true');
async function refreshStatus(applyDefaults = false) {
  const state = await json('/api/status'); lastStatus = state;
  $('#runtime-state').textContent = state.busy ? 'Processing audio' : 'Ready'; $('#runtime-detail').textContent = `${state.device} · ${Object.values(state.models).filter(v => v === 'ready').length} models loaded`;
  $('#runtime-badge').dataset.state = 'ready'; $('#version').textContent = `v${state.version}`;
  $('#queue-state').textContent = state.queue ? `Queue: ${state.queue.active}/${state.queue.capacity} active · ${state.queue.queued} waiting · ${state.queue.running} running · ${state.queue.uploading} uploading` : '';
  $('#system-runtime').textContent = `Sidon ${state.version} · ${state.device}`;
  $('#limits').textContent = `Up to ${state.limits.upload_mb} MB and ${state.limits.audio_seconds / 60} minutes per file. Results and uploaded audio expire after ${state.limits.retention_hours} hours. ${state.limits.active_jobs || 4} active jobs maximum, processed one at a time.`;
  if (applyDefaults) { $('#strength').value = state.settings.strength; $('#strength').oninput(); $('#output-format').value = state.settings.output_format; $('#chunk-seconds').value = state.settings.chunk_seconds; $('#overlap-seconds').value = state.settings.overlap_seconds; modeChanged(); }
  const models = $('#models'); models.replaceChildren();
  for (const [name, status] of Object.entries(state.models)) {
    const row = document.createElement('div'); row.className = 'model-row';
    const label = document.createElement('div'), title = document.createElement('strong'), detail = document.createElement('small');
    title.textContent = name === 'enhance' ? 'Sidon · restoration' : 'MossFormer2 · two speakers'; detail.textContent = state.model_errors[name] || `${status}${name === 'enhance' && state.backends?.[name] ? (state.backends[name] !== 'torchscript-fp32' ? ' · Optimized F32' : ' · Reference F32') : ''}`;
    label.append(title, detail); const button = document.createElement('button'); button.className = 'secondary-button'; button.textContent = status === 'ready' ? 'Unload' : status === 'loading' ? 'Loading…' : 'Load'; button.disabled = state.busy;
    button.onclick = guarded(async () => { button.disabled = true; button.textContent = status === 'ready' ? 'Unloading…' : 'Loading…'; try { await json(`/api/models/${name}`, {method: 'POST', ...jsonOptions({action: status === 'ready' ? 'unload' : 'load'})}); } finally { await refreshStatus(); } });
    row.append(label, button); models.append(row);
  }
}
function clearOutputs() { outputPlayers.forEach(p => p.dispose()); outputPlayers = []; $('#outputs').replaceChildren(); }
function beginJob(jobId = null) {
  clearTimeout(pollTimer); pollGeneration++; activeJob = jobId; clearOutputs();
  $('#download-all').disabled = $('#delete').disabled = $('#cancel').disabled = true;
}
async function showResults(job, isCurrent) {
  clearOutputs();
  for (const track of job.outputs) {
    const endpoint = `/api/jobs/${job.id}/files/${track.file}`;
    const blob = await (await request(endpoint)).blob();
    if (!isCurrent()) return;
    const player = new Player($('#outputs'), track.name === 'enhanced' ? 'Enhanced audio' : track.name === 'converted' ? 'Converted audio' : track.name.replace('_', ' '), {download: guarded(() => download(endpoint, track.file))}); outputPlayers.push(player);
    await player.load(blob, `${track.sample_rate / 1000} kHz · ${job.settings.output_format.toUpperCase()}`);
    if (!isCurrent()) return;
    if (track.samples_over_full_scale) { const warning = document.createElement('p'); warning.className = 'help'; warning.textContent = job.settings.output_format === 'flac' ? 'Some samples exceeded full scale and were clipped by the FLAC encoding. Use float WAV to retain them.' : job.settings.output_format === 'wav' ? 'Some samples exceed full scale. Float WAV preserves them; lower playback volume if needed.' : 'Some source samples exceeded full scale. Choose float WAV to preserve those peaks.'; player.root.append(warning); }
  }
  $('#download-all').disabled = $('#delete').disabled = false;
}
function setRunning(value) { if (value && !running) pollingStarted = performance.now(); running = value; $('#process').disabled = value || !file; $('#cancel').disabled = !value || !activeJob; $('#choose').disabled = $('#mode').disabled = $('#strength').disabled = $('#output-format').disabled = value; }
async function pollJob() {
  clearTimeout(pollTimer); if (!activeJob) return;
  const jobId = activeJob, generation = ++pollGeneration;
  const isCurrent = () => generation === pollGeneration && jobId === activeJob;
  try {
    const job = await json(`/api/jobs/${jobId}`);
    if (!isCurrent()) return;
    message(job.status === 'failed' ? job.error : job.status === 'done' ? 'Loading result audio…' : job.status === 'cancelled' ? 'Cancelled' : job.status === 'queued' ? `Queued · position ${job.queue_position || 1}` : `${job.stage}…`, job.status === 'failed');
    if (job.status === 'done') { setRunning(true); $('#cancel').disabled = true; await showResults(job, isCurrent); if (!isCurrent()) return; setRunning(false); message(`Complete · ${job.elapsed_seconds}s`); await refreshStatus(); }
    else if (['failed','cancelled'].includes(job.status)) { setRunning(false); $('#delete').disabled = false; await refreshStatus(); }
    else { $('#cancel').disabled = !!job.cancel_requested; pollTimer = setTimeout(pollJob, performance.now() - pollingStarted < 10000 ? 250 : 1000); }
  } catch (error) {
    if (!isCurrent()) return;
    if (error.status === 401) { setRunning(false); message(error.message, true); return; }
    if (error.status === 404) { beginJob(); setRunning(false); message('This job was deleted or expired. Submit the audio again.', true); return; }
    message(`${error.message} Retrying job status…`, true); pollTimer = setTimeout(pollJob, 4000);
  }
}
$('#process').onclick = guarded(async () => {
  if (!file || running) return; beginJob(); setRunning(true); message('Uploading audio…');
  const data = new FormData(); data.append('file', file); data.append('mode', $('#mode').value); data.append('strength', $('#strength').value); data.append('output_format', $('#output-format').value);
  try { const job = await json('/api/jobs', {method: 'POST', body: data}); activeJob = job.id; $('#cancel').disabled = false; $('#delete').disabled = $('#download-all').disabled = true; clearOutputs(); await pollJob(); }
  catch (error) { setRunning(false); throw error; }
});
$('#cancel').onclick = guarded(async () => { if (!activeJob) return; $('#cancel').disabled = true; await json(`/api/jobs/${activeJob}/cancel`, {method:'POST'}); await pollJob(); });
$('#download-all').onclick = guarded(() => download(`/api/jobs/${activeJob}/files/results.zip`, 'sidon-results.zip'));
$('#delete').onclick = guarded(async () => { const jobId = activeJob; if (!jobId) return; await json(`/api/jobs/${jobId}`, {method: 'DELETE'}); if (activeJob !== jobId) return; beginJob(); message('Job and server audio deleted.'); });
$('#save-settings').onclick = guarded(async () => { await json('/api/settings', {method: 'PUT', ...jsonOptions({strength: +$('#strength').value, output_format: $('#output-format').value, chunk_seconds: +$('#chunk-seconds').value, overlap_seconds: +$('#overlap-seconds').value})}); message('Server defaults saved.'); await refreshStatus(); });
$('#connect').onclick = guarded(async () => { apiKey = $('#api-key').value; $('#api-key').value = ''; authenticationRequired = false; await refreshStatus(true); message('Connected.'); scheduleGpu(); if (activeJob) await pollJob(); });
async function refreshHistory() {
  const jobs = await json('/api/jobs'); $('#history').replaceChildren();
  for (const job of jobs.slice(0, 25)) {
    const row = document.createElement('div'); row.className = 'history-item'; const text = document.createElement('span'); text.textContent = `${job.filename} · ${job.status}`;
    const button = document.createElement('button'); button.className = 'text-button'; button.textContent = 'Open'; button.disabled = running;
    button.onclick = guarded(async () => {
      if (running) return;
      beginJob(job.id); setRunning(true);
      try {
        changeView(['enhance','convert'].includes(job.mode) ? job.mode : 'separate'); $('#mode').value = job.mode; $('#strength').value = job.settings.strength; $('#strength').oninput(); $('#output-format').value = job.settings.output_format; modeChanged();
        const blob = await (await request(`/api/jobs/${job.id}/files/input.wav`)).blob(); file = new File([blob], job.filename, {type:'audio/wav'}); await inputPlayer.load(blob, job.filename); await pollJob();
      } catch (error) { setRunning(false); throw error; }
    });
    row.append(text, button); $('#history').append(row);
  }
  if (!jobs.length) $('#history').textContent = 'No retained jobs.';
}
$('#refresh-jobs').onclick = guarded(refreshHistory);
const metrics = [['utilization','GPU','%'],['memory_used','VRAM','MiB'],['temperature','Temperature','°C'],['power','Power','W'],['memory_utilization','Memory controller','%'],['fan_speed','Fan','%'],['graphics_clock','Graphics clock','MHz'],['memory_clock','Memory clock','MHz']];
let gpuHistory = {}; try { gpuHistory = JSON.parse(sessionStorage.getItem('sidon-gpu') || '{}'); } catch {}
function drawGpu(data) {
  const cutoff = Date.now() - 600000;
  for (const [index, samples] of Object.entries(data.history)) { const byTime = new Map([...(gpuHistory[index] || []), ...samples].map(s => [s.timestamp, s])); gpuHistory[index] = [...byTime.values()].filter(s => s.timestamp >= cutoff).sort((a,b) => a.timestamp - b.timestamp); }
  try { sessionStorage.setItem('sidon-gpu', JSON.stringify(gpuHistory)); } catch {}
  $('#gpu').replaceChildren();
  if (!data.gpus.length) { $('#gpu').textContent = 'No NVIDIA telemetry available. CPU processing remains available.'; return; }
  for (const gpu of data.gpus) {
    const card = document.createElement('article'); card.className = 'gpu-card'; const heading = document.createElement('h3'); heading.textContent = `GPU ${gpu.index} · ${gpu.name}`; card.append(heading);
    const stats = document.createElement('div'); stats.className = 'gpu-stats'; stats.textContent = `VRAM ${gpu.memory_used ?? '—'} / ${gpu.memory_total ?? '—'} MiB · ${gpu.performance_state ?? '—'} · PCIe ${gpu.pcie_generation ?? '—'} ×${gpu.pcie_width ?? '—'}`; card.append(stats);
    const charts = document.createElement('div'); charts.className = 'charts';
    const now = Date.now(), seconds = +$('#gpu-window').value, history = (gpuHistory[gpu.index] || []).filter(s => s.timestamp >= now - seconds * 1000);
    for (const [field, label, unit] of metrics) {
      if (gpu[field] == null) continue; const metric = document.createElement('div'); metric.className = 'metric'; const title = document.createElement('strong'); title.textContent = label; const value = document.createElement('b'); value.textContent = `${Math.round(gpu[field])} ${unit}`; title.append(value); metric.append(title);
      const samples = history.filter(s => Number.isFinite(s[field])); const maximum = unit === '%' ? 100 : Math.max(1, gpu[field], ...samples.map(s => s[field]));
      const svg = document.createElementNS('http://www.w3.org/2000/svg','svg'); svg.setAttribute('viewBox','0 0 300 80'); svg.setAttribute('preserveAspectRatio','none'); svg.setAttribute('aria-label', `${label} over the last ${seconds} seconds`); svg.setAttribute('role','img');
      const path = document.createElementNS(svg.namespaceURI,'polyline'); path.setAttribute('points', samples.map(s => `${Math.max(0,(s.timestamp - now + seconds * 1000)/(seconds*1000)*300)},${78 - s[field]/maximum*74}`).join(' ')); path.setAttribute('fill','none'); path.setAttribute('stroke','#ff7a1a'); path.setAttribute('stroke-width','2'); path.setAttribute('vector-effect','non-scaling-stroke'); svg.append(path); metric.append(svg); charts.append(metric);
    }
    card.append(charts); $('#gpu').append(card);
  }
}
async function updateGpu() {
  if (authenticationRequired || gpuBusy || view !== 'system' || document.hidden) return; gpuBusy = true;
  try { drawGpu(await json('/api/gpu')); } catch { /* Retain the last valid telemetry during temporary failures. */ }
  finally { gpuBusy = false; if (!authenticationRequired && view === 'system' && !document.hidden) gpuTimer = setTimeout(updateGpu, 1000); }
}
function scheduleGpu() { clearTimeout(gpuTimer); if (view === 'system' && !document.hidden) updateGpu(); }
$('#gpu-window').value = sessionStorage.getItem('sidon-gpu-window') || '60'; $('#gpu-window').onchange = () => { sessionStorage.setItem('sidon-gpu-window', $('#gpu-window').value); scheduleGpu(); };
document.addEventListener('visibilitychange', scheduleGpu); window.addEventListener('pagehide', () => { clearTimeout(gpuTimer); clearTimeout(pollTimer); });
changeView(location.hash.slice(1) || sessionStorage.getItem('sidon-tab') || 'enhance');
refreshStatus(true).catch(error => { if (error.status !== 401) $('#runtime-state').textContent = 'Connection needed'; message(error.message, true); });
setInterval(() => { if (!authenticationRequired && !document.hidden) refreshStatus().catch(() => {}); }, 5000);
