// ProYOLO DefOps • Real-Time Surveillance Telemetry Frontend Logic
// Manages Motion-JPEG video feed, dynamic source switching, WebSocket alert stream,
// and HUD telemetry metrics.

// -------------------------------------------------------------------------
// Global Elements
// -------------------------------------------------------------------------
const elements = {
  videoFeed: document.getElementById('liveVideoFeed'),
  btnWebcam: document.getElementById('btnWebcam'),
  btnUploadVideo: document.getElementById('btnUploadVideo'),
  videoFileInput: document.getElementById('videoFileInput'),
  btnPlayPause: document.getElementById('btnPlayPause'),
  playPauseIcon: document.getElementById('playPauseIcon'),
  playPauseLabel: document.getElementById('playPauseLabel'),
  btnLoopToggle: document.getElementById('btnLoopToggle'),
  activeSourceBadge: document.getElementById('activeSourceBadge'),
  sourceLabelText: document.getElementById('sourceLabelText'),
  uploadStatusText: document.getElementById('uploadStatusText'),
  metricFps: document.getElementById('metricFps'),
  metricLatency: document.getElementById('metricLatency'),
  metricCpu: document.getElementById('metricCpu'),
  metricClean: document.getElementById('metricClean'),
  metricAttacks: document.getElementById('metricAttacks'),
  connectionStatus: document.getElementById('connectionStatus'),
  connectionLabel: document.getElementById('connectionLabel'),
  alertsContainer: document.getElementById('alertsContainer'),
  emptyAlertsNotice: document.getElementById('emptyAlertsNotice'),
  threatBadge: document.getElementById('threatBadge'),
  btnTestAlert: document.getElementById('btnTestAlert'),
  btnAudioToggle: document.getElementById('btnAudioToggle'),
  audioIcon: document.getElementById('audioIcon'),
  btnClearAlerts: document.getElementById('btnClearAlerts'),
  thresholdSlider: document.getElementById('thresholdSlider'),
  thresholdVal: document.getElementById('thresholdVal'),
  snapshotModal: document.getElementById('snapshotModal'),
  modalCloseBtn: document.getElementById('modalCloseBtn'),
  modalImg: document.getElementById('modalImg'),
  modalDetails: document.getElementById('modalDetails')
};

let audioEnabled = true;
let audioCtx = null;
let totalAlertsCount = 0;
let isPaused = false;
let isLooping = true;
let websocket = null;
let reconnectTimeout = null;

// -------------------------------------------------------------------------
// Audio Synthesizer (Radar Chirp for Attacks)
// -------------------------------------------------------------------------
function playAlertPing() {
  if (!audioEnabled) return;
  try {
    if (!audioCtx) {
      const AudioContext = window.AudioContext || window.webkitAudioContext;
      audioCtx = new AudioContext();
    }
    if (audioCtx.state === 'suspended') {
      audioCtx.resume();
    }
    const osc = audioCtx.createOscillator();
    const gain = audioCtx.createGain();
    osc.type = 'sine';
    osc.frequency.setValueAtTime(880, audioCtx.currentTime); // A5
    osc.frequency.exponentialRampToValueAtTime(1760, audioCtx.currentTime + 0.12);
    gain.gain.setValueAtTime(0.3, audioCtx.currentTime);
    gain.gain.exponentialRampToValueAtTime(0.01, audioCtx.currentTime + 0.25);
    osc.connect(gain);
    gain.connect(audioCtx.destination);
    osc.start();
    osc.stop(audioCtx.currentTime + 0.25);
  } catch (e) {
    console.warn('Audio play error:', e);
  }
}

// -------------------------------------------------------------------------
// Threat Feed Management
// -------------------------------------------------------------------------
function addAlertToFeed(alert) {
  if (elements.emptyAlertsNotice) {
    elements.emptyAlertsNotice.style.display = 'none';
  }

  totalAlertsCount++;
  if (elements.threatBadge) {
    elements.threatBadge.innerText = `${totalAlertsCount} Alerts Logged`;
  }
  if (elements.metricAttacks) {
    elements.metricAttacks.innerText = totalAlertsCount;
  }

  const card = document.createElement('div');
  card.className = 'alert-item';
  
  const probPercent = (alert.attack_probability * 100).toFixed(1);
  const confPercent = (alert.confidence * 100).toFixed(1);
  const bboxStr = alert.bbox ? `[${alert.bbox.map(v => Math.round(v)).join(', ')}]` : '[0, 0, 0, 0]';
  const timeStr = alert.created_at ? alert.created_at.replace('T', ' ').slice(0, 19) : new Date().toLocaleTimeString();
  const thumbSrc = alert.thumbnail_b64 || generateSyntheticThumbnail(true);

  const isRecovered = Boolean(alert.recovered_by_inpainting);
  const deltaP = alert.delta_p_obj !== undefined && alert.delta_p_obj !== null ? Number(alert.delta_p_obj).toFixed(2) : '0.00';
  const recoveryBadgeHtml = isRecovered
    ? `<span class="feature-tag highlight" style="background: rgba(6, 182, 212, 0.25); color: #06b6d4; border: 1px solid #06b6d4; font-weight: 700;">🛡 RECOVERED (ΔP: +${deltaP})</span>`
    : `<span class="feature-tag" style="background: rgba(148, 163, 184, 0.15); color: #94a3b8;">RAW TARGET</span>`;

  // Feature breakdown tags
  let featureTagsHtml = recoveryBadgeHtml;
  if (alert.top_features && typeof alert.top_features === 'object') {
    for (const [key, val] of Object.entries(alert.top_features)) {
      const formattedKey = key.replace(/_/g, ' ');
      const formattedVal = typeof val === 'number' ? (val >= 0 ? `+${val.toFixed(2)}` : val.toFixed(2)) : val;
      featureTagsHtml += `<span class="feature-tag highlight">${formattedKey}: ${formattedVal}</span>`;
    }
  } else {
    featureTagsHtml += `
      <span class="feature-tag highlight">HSV Val Entropy: +0.44</span>
      <span class="feature-tag highlight">Confidence Drop: +0.38</span>
      <span class="feature-tag">Laplacian Var: +0.28</span>
    `;
  }

  card.innerHTML = `
    <div class="alert-top">
      <span class="alert-cam-id">${alert.camera_id || 'CAM-01'} • Track #${alert.track_id}</span>
      <div class="alert-prob-pill">
        <span>⚠</span> ${probPercent}% Attacker
      </div>
    </div>
    <div class="alert-content-row">
      <div class="alert-thumbnail-box" title="Click to inspect snapshot">
        <img src="${thumbSrc}" alt="Attacker Snapshot">
        <div class="alert-thumbnail-overlay"></div>
      </div>
      <div class="alert-details">
        <div class="alert-track-info">${isRecovered ? "Actively Recovered Target (Inpainting Mitigation)" : "Attacker Detected (Adversarial Patch)"}</div>
        <div class="alert-coords">Person Confidence: ${confPercent}% • BBox: ${bboxStr}</div>
        <div class="alert-features">${featureTagsHtml}</div>
      </div>
    </div>
    <div class="alert-time">Timestamp: ${timeStr}</div>
  `;

  // Click to open snapshot modal
  const thumbBox = card.querySelector('.alert-thumbnail-box');
  if (thumbBox) {
    thumbBox.addEventListener('click', () => {
      openModal(alert, thumbSrc);
    });
  }

  if (elements.alertsContainer) {
    elements.alertsContainer.insertBefore(card, elements.alertsContainer.firstChild);

    // Cap maximum cards in DOM to 50
    if (elements.alertsContainer.children.length > 51) {
      elements.alertsContainer.removeChild(elements.alertsContainer.lastChild);
    }
  }

  playAlertPing();
}

function openModal(alert, thumbSrc) {
  if (!elements.snapshotModal) return;
  const isRecovered = Boolean(alert.recovered_by_inpainting);
  const deltaP = alert.delta_p_obj !== undefined && alert.delta_p_obj !== null ? Number(alert.delta_p_obj).toFixed(2) : '0.00';

  elements.modalImg.src = thumbSrc;
  elements.modalDetails.innerHTML = `
    <div><strong>Classification:</strong> <span style="color: #ff2a5f; font-weight: 700;">ATTACKER</span> (Adversarial Optical Pattern Detected)</div>
    <div><strong>Detection Status:</strong> ${isRecovered ? `<span style="color: #06b6d4; font-weight: 700;">ACTIVELY RECOVERED</span> (Inpainted Patch, Objectness Recovery ΔP_obj = +${deltaP})` : '<span style="color: #94a3b8; font-weight: 600;">RAW DETECTION</span>'}</div>
    <div><strong>Camera:</strong> ${alert.camera_id || 'CAM-01'}</div>
    <div><strong>Track ID:</strong> #${alert.track_id}</div>
    <div><strong>Attacker Probability:</strong> ${(alert.attack_probability * 100).toFixed(2)}%</div>
    <div><strong>Person Detection Confidence:</strong> ${(alert.confidence * 100).toFixed(2)}%</div>
    <div><strong>Bounding Box Coordinates:</strong> [${alert.bbox ? alert.bbox.join(', ') : '0, 0, 0, 0'}]</div>
    <div><strong>Timestamp:</strong> ${alert.created_at || new Date().toISOString()}</div>
    <div style="margin-top: 8px;"><strong>Forensic Assessment:</strong> High spatial chromatic variance and high-frequency edge gradients detected on human torso, causing sudden confidence attenuation in YOLOv8n backbone. Flagged for optical mitigation and persistence verification.</div>
  `;
  elements.snapshotModal.classList.add('open');
}

if (elements.modalCloseBtn) {
  elements.modalCloseBtn.addEventListener('click', () => {
    elements.snapshotModal.classList.remove('open');
  });
}

if (elements.snapshotModal) {
  elements.snapshotModal.addEventListener('click', (e) => {
    if (e.target === elements.snapshotModal) elements.snapshotModal.classList.remove('open');
  });
}

function generateSyntheticThumbnail(isAttack) {
  const offCanvas = document.createElement('canvas');
  offCanvas.width = 120;
  offCanvas.height = 120;
  const ctx = offCanvas.getContext('2d');
  ctx.fillStyle = '#0a101f';
  ctx.fillRect(0, 0, 120, 120);

  ctx.fillStyle = isAttack ? '#ff2a5f' : '#10b981';
  ctx.beginPath();
  ctx.arc(60, 35, 16, 0, Math.PI * 2);
  ctx.fill();
  ctx.fillRect(40, 56, 40, 50);

  if (isAttack) {
    ctx.fillStyle = '#06b6d4';
    ctx.fillRect(48, 65, 12, 12);
    ctx.fillStyle = '#f59e0b';
    ctx.fillRect(60, 65, 12, 12);
  }
  return offCanvas.toDataURL('image/jpeg', 0.8);
}

// -------------------------------------------------------------------------
// WebSocket Connection & Real-Time Events
// -------------------------------------------------------------------------
function connectWebSocket() {
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  const wsUrl = `${protocol}//${window.location.host}/ws/alerts`;

  try {
    websocket = new WebSocket(wsUrl);

    websocket.onopen = () => {
      if (elements.connectionStatus) elements.connectionStatus.classList.remove('disconnected');
      if (elements.connectionLabel) elements.connectionLabel.innerText = 'WS CONNECTED';
      if (reconnectTimeout) {
        clearTimeout(reconnectTimeout);
        reconnectTimeout = null;
      }
    };

    websocket.onmessage = (event) => {
      try {
        const msg = JSON.parse(event.data);
        handleWsMessage(msg);
      } catch (e) {
        console.error('Error parsing WS message:', e);
      }
    };

    websocket.onclose = () => {
      if (elements.connectionStatus) elements.connectionStatus.classList.add('disconnected');
      if (elements.connectionLabel) elements.connectionLabel.innerText = 'WS RECONNECTING';
      scheduleReconnect();
    };

    websocket.onerror = () => {
      if (elements.connectionStatus) elements.connectionStatus.classList.add('disconnected');
      if (elements.connectionLabel) elements.connectionLabel.innerText = 'WS ERROR';
    };
  } catch (err) {
    scheduleReconnect();
  }
}

function scheduleReconnect() {
  if (!reconnectTimeout) {
    reconnectTimeout = setTimeout(() => {
      reconnectTimeout = null;
      connectWebSocket();
    }, 3000);
  }
}

function handleWsMessage(msg) {
  if (msg.event === 'init_state' || msg.event === 'telemetry_update') {
    if (msg.metrics) updateMetricsUI(msg.metrics);
  } else if (msg.event === 'attack_alert') {
    if (msg.data) addAlertToFeed(msg.data);
    if (msg.metrics) updateMetricsUI(msg.metrics);
  }
}

function updateMetricsUI(m) {
  if (m.fps !== undefined && elements.metricFps) elements.metricFps.innerText = m.fps.toFixed(1);
  if (m.latency_ms !== undefined && elements.metricLatency) elements.metricLatency.innerText = m.latency_ms.toFixed(1);
  if (m.cpu_usage !== undefined && elements.metricCpu) elements.metricCpu.innerText = m.cpu_usage.toFixed(1);
  if (m.clean_detections !== undefined && elements.metricClean) elements.metricClean.innerText = m.clean_detections;
  if (m.flagged_attacks !== undefined && elements.metricAttacks) {
    elements.metricAttacks.innerText = m.flagged_attacks;
    totalAlertsCount = m.flagged_attacks;
    if (elements.threatBadge) {
      elements.threatBadge.innerText = `${totalAlertsCount} Alerts Logged`;
    }
  }
}

// -------------------------------------------------------------------------
// Video Source & Playback Controls
// -------------------------------------------------------------------------
function reloadVideoFeed() {
  if (elements.videoFeed) {
    elements.videoFeed.src = `/api/v1/video-feed?t=${Date.now()}`;
  }
}

if (elements.videoFeed) {
  elements.videoFeed.onerror = () => {
    setTimeout(() => {
      reloadVideoFeed();
    }, 1500);
  };
}

// Switch to live webcam
if (elements.btnWebcam) {
  elements.btnWebcam.addEventListener('click', async () => {
    try {
      const res = await fetch('/api/v1/stream-source', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ source: 0, loop: true })
      });
      if (res.ok) {
        const data = await res.json();
        elements.btnWebcam.classList.add('btn-active');
        if (elements.btnUploadVideo) elements.btnUploadVideo.classList.remove('btn-active');
        if (elements.sourceLabelText) elements.sourceLabelText.innerText = data.source_label || 'Webcam (Live)';
        reloadVideoFeed();
      }
    } catch (e) {
      console.error('Webcam switch error:', e);
    }
  });
}

// File upload trigger
if (elements.btnUploadVideo && elements.videoFileInput) {
  elements.btnUploadVideo.addEventListener('click', () => {
    elements.videoFileInput.click();
  });

  elements.videoFileInput.addEventListener('change', async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    if (elements.uploadStatusText) {
      elements.uploadStatusText.style.display = 'inline-flex';
      elements.uploadStatusText.innerText = `⏳ Uploading ${file.name}...`;
    }

    const formData = new FormData();
    formData.append('file', file);

    try {
      const res = await fetch('/api/v1/upload-video', {
        method: 'POST',
        body: formData
      });

      if (res.ok) {
        const data = await res.json();
        if (elements.uploadStatusText) elements.uploadStatusText.innerText = `✓ Loaded: ${file.name}`;
        elements.btnUploadVideo.classList.add('btn-active');
        if (elements.btnWebcam) elements.btnWebcam.classList.remove('btn-active');
        if (elements.sourceLabelText) elements.sourceLabelText.innerText = `File: ${file.name}`;
        reloadVideoFeed();
        setTimeout(() => {
          if (elements.uploadStatusText) elements.uploadStatusText.style.display = 'none';
        }, 3500);
      } else {
        const err = await res.json();
        alert(`Upload failed: ${err.detail || 'Server error'}`);
        if (elements.uploadStatusText) elements.uploadStatusText.style.display = 'none';
      }
    } catch (err) {
      console.error('Upload error:', err);
      if (elements.uploadStatusText) elements.uploadStatusText.style.display = 'none';
    } finally {
      elements.videoFileInput.value = '';
    }
  });
}

// Play / Pause toggle
if (elements.btnPlayPause) {
  elements.btnPlayPause.addEventListener('click', async () => {
    try {
      const res = await fetch('/api/v1/stream-control', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ action: 'toggle' })
      });
      if (res.ok) {
        const data = await res.json();
        isPaused = data.is_paused;
        if (isPaused) {
          elements.playPauseIcon.innerText = '▶';
          elements.playPauseLabel.innerText = 'Resume';
        } else {
          elements.playPauseIcon.innerText = '⏸';
          elements.playPauseLabel.innerText = 'Pause';
        }
      }
    } catch (e) {
      console.error('Stream pause toggle error:', e);
    }
  });
}

// Loop toggle
if (elements.btnLoopToggle) {
  elements.btnLoopToggle.addEventListener('click', () => {
    isLooping = !isLooping;
    if (isLooping) {
      elements.btnLoopToggle.classList.add('btn-active');
      elements.btnLoopToggle.innerHTML = '<span>🔁</span> Loop: ON';
    } else {
      elements.btnLoopToggle.classList.remove('btn-active');
      elements.btnLoopToggle.innerHTML = '<span>🔁</span> Loop: OFF';
    }
  });
}

// Audio toggle
if (elements.btnAudioToggle) {
  elements.btnAudioToggle.addEventListener('click', () => {
    audioEnabled = !audioEnabled;
    elements.audioIcon.innerText = audioEnabled ? '🔊' : '🔇';
    elements.btnAudioToggle.style.opacity = audioEnabled ? '1' : '0.6';
  });
}

// Clear alerts feed
if (elements.btnClearAlerts) {
  elements.btnClearAlerts.addEventListener('click', () => {
    elements.alertsContainer.innerHTML = '';
    if (elements.emptyAlertsNotice) {
      elements.emptyAlertsNotice.style.display = 'flex';
      elements.alertsContainer.appendChild(elements.emptyAlertsNotice);
    }
  });
}

// Trigger test alert
if (elements.btnTestAlert) {
  elements.btnTestAlert.addEventListener('click', async () => {
    const sampleThumb = generateSyntheticThumbnail(true);
    const randomTrack = Math.floor(Math.random() * 80) + 10;
    const testPayload = {
      camera_id: 'CAM-01',
      track_id: randomTrack,
      confidence: 0.52,
      attack_probability: 0.94,
      bbox: [180.0, 120.0, 360.0, 480.0],
      thumbnail_b64: sampleThumb,
      top_features: {
        hsv_val_entropy: 0.44,
        confidence_drop: 0.38,
        laplacian_var: 0.29,
        flicker_rate: 0.22
      }
    };

    try {
      const res = await fetch('/api/v1/alerts', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(testPayload)
      });
      if (!res.ok) console.error('Failed to post test alert:', await res.text());
    } catch (err) {
      console.error('Trigger alert error:', err);
    }
  });
}

// Threshold slider interaction
if (elements.thresholdSlider && elements.thresholdVal) {
  elements.thresholdSlider.addEventListener('input', (e) => {
    elements.thresholdVal.innerText = e.target.value;
  });
}

// -------------------------------------------------------------------------
// Initial Bootstrap
// -------------------------------------------------------------------------
async function loadInitialData() {
  try {
    const metricsRes = await fetch('/api/v1/metrics');
    if (metricsRes.ok) {
      const metrics = await metricsRes.json();
      updateMetricsUI(metrics);
    }

    const alertsRes = await fetch('/api/v1/alerts?limit=15');
    if (alertsRes.ok) {
      const data = await alertsRes.json();
      if (data.items && data.items.length > 0) {
        data.items.reverse().forEach(a => addAlertToFeed(a));
      }
    }

    const statusRes = await fetch('/api/v1/stream-status');
    if (statusRes.ok) {
      const statusData = await statusRes.json();
      if (statusData.source_label && elements.sourceLabelText) {
        elements.sourceLabelText.innerText = statusData.source_label;
      }
    }
  } catch (err) {
    console.warn('Initial data load error:', err);
  }
}

window.addEventListener('DOMContentLoaded', () => {
  connectWebSocket();
  loadInitialData();
});
