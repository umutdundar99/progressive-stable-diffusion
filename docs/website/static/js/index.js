/* =====================================================
   DADD Demo – interactive disease progression slider
   ===================================================== */

/* ── Patient catalogue ─────────────────────────────────
   Keys = source MES level (0-3).
   Values = patient names matching the inference output dirs.
   ─────────────────────────────────────────────────────── */
const PATIENTS = {
  0: [
    'UC_patient_154_10',
    'UC_patient_25_34',
    'UC_patient_323_56',
    'UC_patient_493_18',
    'UC_patient_502_11',
    'UC_patient_8_12',
    'UC_patient_9_6',
  ],
  1: [
    'UC_patient_10_3',
    'UC_patient_165_7',
    'UC_patient_19_19',
    'UC_patient_22_39',
    'UC_patient_37_57',
    'UC_patient_380_50',
    'UC_patient_514_41',
  ],
  2: [
    'UC_patient_24_23',
    'UC_patient_261_17',
    'UC_patient_449_10',
    'UC_patient_463_46',
    'UC_patient_60_17',
    'UC_patient_8_4',
    'UC_patient_93_43',
  ],
  3: [
    'UC_patient_196_21',
    'UC_patient_199_47',
    'UC_patient_226_42',
    'UC_patient_24_27',
    'UC_patient_268_17',
    'UC_patient_60_3',
    'UC_patient_97_3',
  ],
};

/* 31 steps: 0.00, 0.10, …, 3.00 */
const MES_STEPS = 31;
const MES_VALUES = Array.from({ length: MES_STEPS }, (_, i) =>
  (i * 0.1).toFixed(2)
);

/* MES level display names */
const MES_NAMES = { 0: 'Remission', 1: 'Mild', 2: 'Moderate', 3: 'Severe' };

/* MES level accent colours (matching CSS vars) */
const MES_COLORS = {
  0: '#27ae60',
  1: '#f39c12',
  2: '#e67e22',
  3: '#c0392b',
};

/* ── State ─────────────────────────────────────────── */
let currentMes   = 0;     // source MES level (tab)
let currentPatient = PATIENTS[0][0];
let animTimer    = null;
let animDir      = 1;     // +1 forward, -1 reverse
let isPlaying    = false;

/* ── DOM refs ──────────────────────────────────────── */
const slider         = document.getElementById('mesSlider');
const generatedImg   = document.getElementById('generatedImg');
const sourceImg      = document.getElementById('sourceImg');
const sourceBadge    = document.getElementById('sourceBadge');
const mesBarFill     = document.getElementById('mesBarFill');
const mesBarThumb    = document.getElementById('mesBarThumb');
const mesValueLabel  = document.getElementById('mesValueLabel');
const loadingOverlay = document.getElementById('loadingOverlay');
const thumbnailStrip = document.getElementById('thumbnailStrip');
const btnPlayPause   = document.getElementById('btnPlayPause');
const playIcon       = document.getElementById('playIcon');
const playLabel      = document.getElementById('playLabel');
const btnReverse     = document.getElementById('btnReverse');
const pingpong       = document.getElementById('pingpong');

/* ── Path helpers ──────────────────────────────────── */
function inferenceBase() {
  return `static/images/inference/source_${currentMes}/${currentPatient}`;
}

function mesImagePath(stepIdx) {
  const val = MES_VALUES[stepIdx];
  const idx = String(stepIdx).padStart(2, '0');
  return `${inferenceBase()}/mes_${val}_${idx}.png`;
}

function sourcePath() {
  return `${inferenceBase()}/structure_reference.png`;
}

/* ── Image update ──────────────────────────────────── */
function updateGeneratedImage(stepIdx) {
  const path = mesImagePath(stepIdx);
  const val  = parseFloat(MES_VALUES[stepIdx]);
  const pct  = (val / 3) * 100;

  /* bar + thumb */
  mesBarFill.style.width    = pct + '%';
  mesBarThumb.style.left    = pct + '%';
  mesValueLabel.textContent = val.toFixed(2);

  /* swap image */
  loadingOverlay.classList.add('is-visible');
  const img = new Image();
  img.onload = () => {
    generatedImg.src = path;
    loadingOverlay.classList.remove('is-visible');
  };
  img.onerror = () => {
    generatedImg.src = '';
    loadingOverlay.classList.remove('is-visible');
  };
  img.src = path;
}

/* ── Preload all 31 frames for smooth animation ──── */
function preloadFrames() {
  for (let i = 0; i < MES_STEPS; i++) {
    const img = new Image();
    img.src = mesImagePath(i);
  }
}

/* ── Patient switch ─────────────────────────────────── */
function switchPatient(patient, mes) {
  stopAnimation();
  currentMes     = mes;
  currentPatient = patient;

  /* reset slider to source MES position */
  const startStep = Math.round(mes * 10);
  slider.value = startStep;

  /* source image */
  sourceImg.src = sourcePath();
  sourceBadge.textContent = `MES ${mes} · ${MES_NAMES[mes]}`;

  /* update thumbnail selection */
  document.querySelectorAll('.thumb-item').forEach(el => {
    el.classList.toggle('is-selected', el.dataset.patient === patient);
  });

  preloadFrames();
  updateGeneratedImage(startStep);
}

/* ── Thumbnail strip ────────────────────────────────── */
function buildThumbnails(mes) {
  thumbnailStrip.innerHTML = '';
  PATIENTS[mes].forEach(patient => {
    const div = document.createElement('div');
    div.className = 'thumb-item' + (patient === currentPatient ? ' is-selected' : '');
    div.dataset.patient = patient;

    const img = document.createElement('img');
    img.src = `static/images/inference/source_${mes}/${patient}/structure_reference.png`;
    img.alt = patient;
    img.title = patient.replace('UC_patient_', 'Patient ');

    div.appendChild(img);
    div.addEventListener('click', () => switchPatient(patient, mes));
    thumbnailStrip.appendChild(div);
  });
}

/* ── Tab switching ──────────────────────────────────── */
document.querySelectorAll('#mesTabs li').forEach(tab => {
  tab.addEventListener('click', () => {
    const mes = parseInt(tab.dataset.mes, 10);

    /* update active tab */
    document.querySelectorAll('#mesTabs li').forEach(t =>
      t.classList.remove('is-active')
    );
    tab.classList.add('is-active');

    /* rebuild thumbnails and switch to first patient */
    buildThumbnails(mes);
    switchPatient(PATIENTS[mes][0], mes);
  });
});

/* ── Slider ─────────────────────────────────────────── */
slider.addEventListener('input', () => {
  stopAnimation();
  updateGeneratedImage(parseInt(slider.value, 10));
});

/* ── Animation ──────────────────────────────────────── */
function stopAnimation() {
  if (animTimer) clearInterval(animTimer);
  animTimer = null;
  isPlaying = false;
  playIcon.className  = 'fas fa-play';
  playLabel.textContent = 'Animate';
}

function startAnimation() {
  isPlaying = true;
  playIcon.className  = 'fas fa-pause';
  playLabel.textContent = 'Pause';

  animTimer = setInterval(() => {
    let step = parseInt(slider.value, 10) + animDir;

    if (step >= MES_STEPS) {
      if (pingpong.checked) {
        animDir = -1;
        step = MES_STEPS - 2;
      } else {
        step = 0;
      }
    } else if (step < 0) {
      if (pingpong.checked) {
        animDir = 1;
        step = 1;
      } else {
        step = MES_STEPS - 1;
      }
    }

    slider.value = step;
    updateGeneratedImage(step);
  }, 120);
}

btnPlayPause.addEventListener('click', () => {
  if (isPlaying) stopAnimation();
  else startAnimation();
});

btnReverse.addEventListener('click', () => {
  animDir = -animDir;
  if (!isPlaying) {
    let step = parseInt(slider.value, 10) + animDir;
    step = Math.max(0, Math.min(MES_STEPS - 1, step));
    slider.value = step;
    updateGeneratedImage(step);
  }
});

/* ── BibTeX copy ────────────────────────────────────── */
function copyBibtex() {
  const text = document.getElementById('bibtexBlock').textContent;
  navigator.clipboard.writeText(text).then(() => {
    const label = document.getElementById('copyLabel');
    label.textContent = 'Copied!';
    setTimeout(() => (label.textContent = 'Copy BibTeX'), 2000);
  });
}
window.copyBibtex = copyBibtex;

/* ── Init ──────────────────────────────────────────── */
(function init() {
  buildThumbnails(0);
  switchPatient(PATIENTS[0][0], 0);
})();
