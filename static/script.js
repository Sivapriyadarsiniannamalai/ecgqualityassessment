/* ═══════════════════════════════════════════════════════════
   ECGPulse — Script
   ═══════════════════════════════════════════════════════════ */

// ── DOM References ────────────────────────────────────────
const dropArea       = document.getElementById('drop-area');
const fileElem       = document.getElementById('fileElem');
const browseBtn      = document.getElementById('browse-btn');
const fileInfo       = document.getElementById('file-info');
const filenameDisp   = document.getElementById('filename-display');
const filesizeDisp   = document.getElementById('filesize-display');
const analyzeBtn     = document.getElementById('analyze-btn');
const resultsSection = document.getElementById('results-section');
const loader         = document.getElementById('loader');
const windowsContainer = document.getElementById('windows-container');

let selectedFile = null;

// ── Background ECG Canvas ─────────────────────────────────
(function initBgCanvas() {
    const canvas = document.getElementById('ecg-bg-canvas');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    let w, h, offset = 0;

    function resize() {
        w = canvas.width  = canvas.offsetWidth;
        h = canvas.height = canvas.offsetHeight;
    }
    window.addEventListener('resize', resize);
    resize();

    function ecgY(x) {
        const t = ((x + offset) % 200) / 200;
        const mid = h / 2;
        if (t > 0.38 && t < 0.42) return mid - h * 0.35;
        if (t > 0.42 && t < 0.46) return mid + h * 0.18;
        if (t > 0.46 && t < 0.50) return mid - h * 0.08;
        return mid + Math.sin(t * Math.PI * 2) * 3;
    }

    function draw() {
        ctx.clearRect(0, 0, w, h);
        ctx.beginPath();
        ctx.strokeStyle = 'rgba(6,182,212,0.5)';
        ctx.lineWidth = 1.5;
        for (let x = 0; x <= w; x += 2) {
            const y = ecgY(x);
            x === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
        }
        ctx.stroke();
        offset += 1.2;
        requestAnimationFrame(draw);
    }
    draw();
})();

// ── Dropzone mouse-follow glow ────────────────────────────
dropArea.addEventListener('mousemove', (e) => {
    const rect = dropArea.getBoundingClientRect();
    dropArea.style.setProperty('--mx', ((e.clientX - rect.left) / rect.width * 100) + '%');
    dropArea.style.setProperty('--my', ((e.clientY - rect.top)  / rect.height * 100) + '%');
});

// ── Drag & Drop ───────────────────────────────────────────
['dragenter','dragover','dragleave','drop'].forEach(evt =>
    dropArea.addEventListener(evt, e => { e.preventDefault(); e.stopPropagation(); }, false));

['dragenter','dragover'].forEach(evt =>
    dropArea.addEventListener(evt, () => dropArea.classList.add('drag-over'), false));

['dragleave','drop'].forEach(evt =>
    dropArea.addEventListener(evt, () => dropArea.classList.remove('drag-over'), false));

dropArea.addEventListener('drop', e => handleFiles(e.dataTransfer.files), false);
browseBtn.addEventListener('click', () => fileElem.click());
fileElem.addEventListener('change', function() { handleFiles(this.files); });

function handleFiles(files) {
    if (!files.length) return;
    selectedFile = files[0];
    filenameDisp.textContent = selectedFile.name;
    filesizeDisp.textContent = formatBytes(selectedFile.size);
    fileInfo.classList.remove('hidden');
}

function formatBytes(bytes) {
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1048576) return (bytes / 1024).toFixed(1) + ' KB';
    return (bytes / 1048576).toFixed(1) + ' MB';
}

// ── Upload & Analyze ──────────────────────────────────────
analyzeBtn.addEventListener('click', async () => {
    if (!selectedFile) return;

    const form = new FormData();
    form.append('file', selectedFile);

    loader.classList.remove('hidden');
    resultsSection.classList.add('hidden');

    // Update status pill
    const pill = document.getElementById('status-pill');
    pill.innerHTML = '<span class="status-dot" style="background:var(--amber);"></span><span>Analyzing…</span>';
    pill.style.background = 'rgba(245,158,11,0.1)';
    pill.style.borderColor = 'rgba(245,158,11,0.2)';
    pill.querySelector('.status-dot').style.background = 'var(--amber)';
    pill.querySelector('span:last-child').style.color = 'var(--amber)';

    try {
        const res  = await fetch('/api/upload', { method: 'POST', body: form });
        const data = await res.json();
        if (res.ok) {
            displayResults(data);
            pill.innerHTML = '<span class="status-dot"></span><span>Analysis Complete</span>';
            pill.style.background = 'rgba(39,174,96,0.1)';
            pill.style.borderColor = 'rgba(39,174,96,0.2)';
        } else {
            alert('Error: ' + data.error);
            pill.innerHTML = '<span class="status-dot" style="background:var(--noisy)"></span><span>Error</span>';
            pill.style.background = 'rgba(231,76,60,0.1)';
        }
    } catch (err) {
        console.error(err);
        alert('Connection error.');
    } finally {
        loader.classList.add('hidden');
    }
});

// ── Render Results ────────────────────────────────────────
function displayResults(data) {
    resultsSection.classList.remove('hidden');

    const cleanCount = data.results.filter(r => r.label === 'CLEAN').length;
    const noisyCount = data.results.filter(r => r.label === 'NOISY').length;
    const total      = data.total_windows;
    const qualityPct = Math.round((cleanCount / total) * 100);

    // Stats
    document.getElementById('stat-clean').textContent = cleanCount;
    document.getElementById('stat-noisy').textContent = noisyCount;
    document.getElementById('stat-total').textContent = total;
    document.getElementById('result-filename').textContent = data.filename;

    // Gauge animation
    animateGauge(qualityPct);

    // Signal Map
    const mapContainer = document.getElementById('signal-map-container');
    if (data.map_image) {
        mapContainer.innerHTML = `<img src="/outputs/${data.map_image}" alt="Signal Quality Map">`;
    } else {
        mapContainer.innerHTML = '<p style="color:var(--text-3)">Map not generated.</p>';
    }

    // Window Cards
    windowsContainer.innerHTML = '';
    data.results.forEach((r, idx) => {
        const isClean  = r.label === 'CLEAN';
        const card = document.createElement('div');
        card.className = `window-card ${isClean ? 'clean-card' : 'noisy-card'}`;
        card.style.animationDelay = `${idx * 0.06}s`;

        const barColor = isClean ? 'var(--clean)' : 'var(--noisy)';
        const badge    = isClean ? 'badge-clean' : 'badge-noisy';
        const icon     = isClean ? '✓' : '✗';

        card.innerHTML = `
            <div class="win-header">
                <span class="win-id">Window #${r.window}</span>
                <span class="badge ${badge}">${icon} ${r.label}</span>
            </div>
            <div class="win-time">${r.start_s}s — ${r.end_s}s</div>
            <div class="conf-row">
                <span class="conf-label">Confidence</span>
                <span class="conf-value" style="color:${barColor}">${r.confidence}%</span>
            </div>
            <div class="conf-bar-bg">
                <div class="conf-bar-fill" style="width:0%; background:${barColor}"></div>
            </div>
        `;
        windowsContainer.appendChild(card);

        // Animate bar after paint
        requestAnimationFrame(() => {
            requestAnimationFrame(() => {
                card.querySelector('.conf-bar-fill').style.width = r.confidence + '%';
            });
        });
    });

    resultsSection.scrollIntoView({ behavior: 'smooth' });
}

// ── Gauge Animation ───────────────────────────────────────
function animateGauge(pct) {
    const arc   = document.getElementById('gauge-arc');
    const label = document.getElementById('gauge-label');
    const value = document.getElementById('gauge-value');
    const maxDash = 251.2; // half-circle circumference
    const target  = (pct / 100) * maxDash;

    let current = 0;
    const step = () => {
        current += (target - current) * 0.06;
        if (Math.abs(target - current) < 0.5) current = target;
        arc.setAttribute('stroke-dasharray', `${current} ${maxDash}`);
        value.textContent = Math.round((current / maxDash) * 100) + '%';
        if (current < target) requestAnimationFrame(step);
    };
    requestAnimationFrame(step);

    if (pct >= 70)      label.textContent = 'Good Quality';
    else if (pct >= 40) label.textContent = 'Moderate Quality';
    else                label.textContent = 'Poor Quality';
}
