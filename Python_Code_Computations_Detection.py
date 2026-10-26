"""
Real-time ECG / EEG / GSR dashboard, laid out to match the sketch:

    -----------------------------------------------------------------
    | [ ECG chart ]              |  Calibrating:                    |
    | HRV, Mean Heart Rate: ...  |  <status>                        |
    |-----------------------------|-----------------------------------|
    | [ EEG chart ]               |  Time Elapsed                    |
    | State of Emotion: ...       |  <timer / progress bar>          |
    |-----------------------------|-----------------------------------|
    | [ GSR chart ]                |  Instructions (What to do)       |
    | Arousal State: ...           |  <phase instructions>            |
    -----------------------------------------------------------------
                                     Overall State: ...

Charts render as a real sweeping-oscilloscope trace: a fixed-size
buffer that the beam writes into and wraps around, overwriting old
data in place, with a small moving blank gap ahead of the write head
(the classic hospital-monitor look) instead of a scrolling strip
chart. ECG peak markers are recomputed every frame directly from
what's currently in that buffer, so a marker disappears the instant
the beam overwrites the sample it was on - nothing goes stale, and
pressing R wipes both the trace and the markers together.

Keyboard controls (click on the plot window first so it has focus):
    C -> start/restart calibration (RELAX phase, then FOCUS phase)
    R -> restart everything (clears data, goes back to idle)
    Q -> quit the program

Serial input format (one line per sample, 3 comma-separated values):
    <GSR_value>,<ECG_value>,<EEG_value>

- value1 (GSR)  -> skin conductance / arousal
- value2 (ECG)  -> heart signal -> HR + HRV
- value3 (EEG)  -> brain signal -> band powers -> attention/meditation/drowsy
                   + an "EEG-HRV" style variability index

Calibration:
    Pressing S runs a short RELAX phase followed by a short FOCUS phase.
    During both phases the EEG band-power ratios and the GSR level are
    recorded to build a personal baseline (mean/std). After calibration
    finishes, live classification (EEG mental state, GSR arousal, and
    the fused "Overall State") is computed relative to *your* baseline
    instead of generic fixed thresholds.
"""

import time
import collections

import matplotlib.animation as animation
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import serial
from scipy.signal import butter, filtfilt, detrend, find_peaks

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
SERIAL_PORT = "COM5"
BAUD_RATE = 115200
SAMPLE_RATE = 184                      # Hz, shared by all 3 channels
WINDOW_SECONDS = 5
WINDOW_SIZE = SAMPLE_RATE * WINDOW_SECONDS   # samples per analysis window / sweep lap

SWEEP_GAP_SAMPLES = 10   # width of the moving blank "beam" gap ahead of the write head

CALIB_RELAX_SECONDS = 20
CALIB_FOCUS_SECONDS = 20

BANDS = {
    "delta":      (0.5, 4.0),
    "theta":      (4.0, 8.0),
    "low_alpha":  (8.0, 10.0),
    "high_alpha": (10.0, 12.0),
    "low_beta":   (12.0, 20.0),
    "high_beta":  (20.0, 30.0),
    "low_gamma":  (30.0, 40.0),
    "mid_gamma":  (40.0, 50.0),
}
EPS = 1e-9

BANDPASS_LOW_HZ = 0.5
BANDPASS_HIGH_HZ = 45.0
BANDPASS_ORDER = 4

EEG_HISTORY_LEN = 12   # rolling window count used for the EEG-HRV index

MODE_IDLE = "IDLE"
MODE_CALIB_RELAX = "CALIB_RELAX"
MODE_CALIB_FOCUS = "CALIB_FOCUS"
MODE_RUNNING = "RUNNING"

# ----------------------------------------------------------------------
# Serial setup
# ----------------------------------------------------------------------
try:
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
except Exception as e:
    print(f"Error opening serial port: {e}")
    raise SystemExit(1)


# ----------------------------------------------------------------------
# Sweeping oscilloscope buffer
# ----------------------------------------------------------------------
class SweepBuffer:
    """
    Fixed-size circular buffer that mimics a real oscilloscope / patient
    monitor beam: new samples are written at an advancing position and
    wrap back to the start, overwriting old data in place. A small
    blank gap is kept just ahead of the write head so the trace shows
    a visible moving "beam" instead of instantly overwriting old data
    with no seam.
    """

    def __init__(self, size, gap=SWEEP_GAP_SAMPLES):
        self.size = size
        self.gap = gap
        self.data = np.full(size, np.nan)
        self.pos = 0

    def push(self, value):
        self.data[self.pos] = value
        for i in range(1, self.gap + 1):
            self.data[(self.pos + i) % self.size] = np.nan
        self.pos = (self.pos + 1) % self.size

    def reset(self):
        self.data[:] = np.nan
        self.pos = 0


sweep_ecg = SweepBuffer(WINDOW_SIZE)
sweep_eeg = SweepBuffer(WINDOW_SIZE)
sweep_gsr = SweepBuffer(WINDOW_SIZE)
SWEEP_X = np.arange(WINDOW_SIZE)

# ----------------------------------------------------------------------
# Figure layout (mirrors the sketch: 3 stacked charts+labels on the
# left, a sectioned sidebar - Calibrating / Time Elapsed / Instructions
# / Overall State - on the right)
# ----------------------------------------------------------------------
fig = plt.figure(figsize=(13, 8.5))
outer = fig.add_gridspec(
    nrows=1, ncols=2, width_ratios=[3, 1.15],
    wspace=0.06, left=0.05, right=0.98, top=0.96, bottom=0.05,
)

left = outer[0, 0].subgridspec(
    nrows=6, ncols=1,
    height_ratios=[1, 0.32, 1, 0.32, 1, 0.32],
    hspace=0.45,
)

LABEL_FONTSIZE = 13
TITLE_FONTSIZE = 16

# --- ECG ---
ax_ecg = fig.add_subplot(left[0, 0])
ax_ecg.set_title("ECG", fontsize=TITLE_FONTSIZE, fontweight="bold")
ax_ecg.set_xlim(0, WINDOW_SIZE)
ax_ecg.set_ylim(0, 1024)
ax_ecg.set_xticks([])
for spine in ax_ecg.spines.values():
    spine.set_linewidth(1.4)
line_ecg, = ax_ecg.plot([], [], color="tab:red", linewidth=1.3)
peaks_marker, = ax_ecg.plot([], [], "go", markersize=6)

ax_ecg_label = fig.add_subplot(left[1, 0])
ax_ecg_label.axis("off")
ecg_label_text = ax_ecg_label.text(
    0.0, 0.5, "HRV, Mean Heart Rate: --", fontsize=LABEL_FONTSIZE,
    fontweight="bold", va="center", transform=ax_ecg_label.transAxes,
)

# --- EEG ---
ax_eeg = fig.add_subplot(left[2, 0])
ax_eeg.set_title("EEG", fontsize=TITLE_FONTSIZE, fontweight="bold")
ax_eeg.set_xlim(0, WINDOW_SIZE)
ax_eeg.set_ylim(0, 1024)
ax_eeg.set_xticks([])
for spine in ax_eeg.spines.values():
    spine.set_linewidth(1.4)
line_eeg, = ax_eeg.plot([], [], color="tab:orange", linewidth=1.3)

ax_eeg_label = fig.add_subplot(left[3, 0])
ax_eeg_label.axis("off")
eeg_label_text = ax_eeg_label.text(
    0.0, 0.5, "State of Emotion: --", fontsize=LABEL_FONTSIZE,
    fontweight="bold", va="center", transform=ax_eeg_label.transAxes,
)

# --- GSR ---
ax_gsr = fig.add_subplot(left[4, 0])
ax_gsr.set_title("GSR", fontsize=TITLE_FONTSIZE, fontweight="bold")
ax_gsr.set_xlim(0, WINDOW_SIZE)
ax_gsr.set_ylim(0, 1024)
ax_gsr.set_xticks([])
for spine in ax_gsr.spines.values():
    spine.set_linewidth(1.4)
line_gsr, = ax_gsr.plot([], [], color="tab:blue", linewidth=1.3)

ax_gsr_label = fig.add_subplot(left[5, 0])
ax_gsr_label.axis("off")
gsr_label_text = ax_gsr_label.text(
    0.0, 0.5, "Arousal State: --", fontsize=LABEL_FONTSIZE,
    fontweight="bold", va="center", transform=ax_gsr_label.transAxes,
)

# --- Right sidebar (Calibrating / Time Elapsed / Instructions / Overall State) ---
ax_side = fig.add_subplot(outer[0, 1])
ax_side.set_xlim(0, 1)
ax_side.set_ylim(0, 1)
ax_side.axis("off")

ax_side.add_patch(patches.Rectangle(
    (0, 0), 1, 1, fill=False, edgecolor="black",
    linewidth=1.6, transform=ax_side.transAxes, clip_on=False,
))
DIVIDER_YS = [0.775, 0.535, 0.30]
for y in DIVIDER_YS:
    ax_side.axhline(y, color="gray", linewidth=1.0)

HEADER_FONTSIZE = 14
VALUE_FONTSIZE = 12.5

# Section 1: Calibrating
ax_side.text(0.5, 0.935, "Calibrating:", ha="center", fontsize=HEADER_FONTSIZE,
             fontweight="bold")
calib_status_text = ax_side.text(
    0.5, 0.855, "IDLE", ha="center", va="center", fontsize=VALUE_FONTSIZE + 1,
    fontweight="bold", color="dimgray",
)

# Section 2: Time Elapsed (+ progress bar)
ax_side.text(0.5, 0.735, "Time Elapsed", ha="center", fontsize=HEADER_FONTSIZE,
             fontweight="bold")
time_value_text = ax_side.text(
    0.5, 0.665, "--", ha="center", va="center", fontsize=VALUE_FONTSIZE + 2,
    fontweight="bold",
)
progress_track = patches.Rectangle((0.08, 0.575), 0.84, 0.055, fill=True,
                                    facecolor="whitesmoke", edgecolor="gray",
                                    transform=ax_side.transAxes)
progress_fill = patches.Rectangle((0.08, 0.575), 0.0, 0.055, fill=True,
                                   facecolor="tab:orange", edgecolor="none",
                                   transform=ax_side.transAxes)
ax_side.add_patch(progress_track)
ax_side.add_patch(progress_fill)

# Section 3: Instructions
ax_side.text(0.5, 0.485, "Instructions", ha="center", fontsize=HEADER_FONTSIZE,
             fontweight="bold")
ax_side.text(0.5, 0.44, "(What to do)", ha="center", fontsize=VALUE_FONTSIZE - 1,
             color="dimgray", style="italic")
instructions_text = ax_side.text(
    0.5, 0.365, "Press C to start\ncalibration", ha="center", va="center",
    fontsize=VALUE_FONTSIZE, wrap=True,
)

# Section 4: Overall State
ax_side.text(0.5, 0.245, "Overall State:", ha="center", fontsize=HEADER_FONTSIZE,
             fontweight="bold")
overall_value_text = ax_side.text(
    0.5, 0.15, "--", ha="center", va="center", fontsize=VALUE_FONTSIZE + 3,
    fontweight="bold", color="tab:blue", wrap=True,
)

keys_hint_text = ax_side.text(
    0.5, 0.025, "[C] Calibrate    [R] Reset    [Q] Quit",
    ha="center", fontsize=9.5, color="dimgray",
)

# ----------------------------------------------------------------------
# Rolling / session state (linear, non-wrapping - used for analysis only)
# ----------------------------------------------------------------------
stor1 = collections.deque([0] * WINDOW_SIZE, maxlen=WINDOW_SIZE)   # GSR
stor2 = collections.deque([0] * WINDOW_SIZE, maxlen=WINDOW_SIZE)   # ECG
stor3 = collections.deque([0] * WINDOW_SIZE, maxlen=WINDOW_SIZE)   # EEG
samples_since_window = 0

mode = MODE_IDLE
calib_phase_start = None
baseline = None   # dict of mean/std for eeg metrics + gsr, set once calibrated

calib_eeg_attention, calib_eeg_meditation = [], []
calib_eeg_drowsy, calib_eeg_high_beta = [], []
calib_gsr_means = []

eeg_attention_hist = collections.deque(maxlen=EEG_HISTORY_LEN)
eeg_meditation_hist = collections.deque(maxlen=EEG_HISTORY_LEN)


# ----------------------------------------------------------------------
# EEG helpers
# ----------------------------------------------------------------------
def preprocess_eeg_window(raw_window, sample_rate):
    x = np.asarray(raw_window, dtype=np.float64)
    x = detrend(x, type="linear")
    nyquist = 0.5 * sample_rate
    b, a = butter(BANDPASS_ORDER,
                  [BANDPASS_LOW_HZ / nyquist, BANDPASS_HIGH_HZ / nyquist],
                  btype="band")
    return filtfilt(b, a, x)


def eeg_band_powers(cleaned_window, sample_rate):
    n = len(cleaned_window)
    x = cleaned_window * np.hamming(n)
    spectrum = np.fft.rfft(x)
    power = np.abs(spectrum) ** 2
    freqs = np.fft.rfftfreq(n, d=1.0 / sample_rate)
    return {name: float(np.sum(power[(freqs >= lo) & (freqs < hi)]))
            for name, (lo, hi) in BANDS.items()}


def eeg_derived_metrics(powers):
    alpha = powers["low_alpha"] + powers["high_alpha"]
    beta = powers["low_beta"] + powers["high_beta"]
    theta = powers["theta"]
    total = sum(powers.values()) + EPS
    return {
        "attention": beta / (alpha + theta + EPS),
        "meditation": alpha / (beta + EPS),
        "drowsy": (powers["delta"] + powers["theta"]) / total,
        "high_beta": powers["high_beta"] / total,
    }


def zscore(value, mean, std):
    std = std if std > EPS else EPS
    return (value - mean) / std


def classify_eeg_state(metrics, base):
    z_att = zscore(metrics["attention"], base["attention_mean"], base["attention_std"])
    z_med = zscore(metrics["meditation"], base["meditation_mean"], base["meditation_std"])
    z_drowsy = zscore(metrics["drowsy"], base["drowsy_mean"], base["drowsy_std"])
    z_hbeta = zscore(metrics["high_beta"], base["high_beta_mean"], base["high_beta_std"])

    if z_drowsy > 1.0:
        state = "Drowsy / Low Alertness"
    elif z_hbeta > 1.0 and z_med < -0.5:
        state = "Stressed / Anxious"
    elif z_att > 0.5 and z_med < 0:
        state = "Focused / Concentrating"
    elif z_med > 0.5 and z_att < 0:
        state = "Relaxed / Calm"
    elif z_att > 0.5 and z_med > 0.5:
        state = "Calm Focus (Flow-like)"
    elif abs(z_att) < 0.5 and abs(z_med) < 0.5:
        state = "Neutral / Baseline"
    else:
        state = "Mixed / Transitional"
    return state, z_att, z_med


def eeg_hrv_index():
    if len(eeg_attention_hist) < 4:
        return None
    return float(np.std(eeg_attention_hist) + np.std(eeg_meditation_hist))


def classify_gsr_arousal(gsr_mean, base):
    z_gsr = zscore(gsr_mean, base["gsr_mean"], base["gsr_std"])
    if z_gsr > 1.0:
        return "HIGH", z_gsr
    if z_gsr > 0.25:
        return "MODERATE", z_gsr
    return "LOW", z_gsr


def overall_emotion(arousal, hrv, eeg_state, eeg_variability):
    hrv_low = hrv is not None and hrv < 5
    hrv_high = hrv is not None and hrv >= 5
    var_high = eeg_variability is not None and eeg_variability > 1.0

    if eeg_state == "Drowsy / Low Alertness" and arousal == "LOW":
        return "Drowsy / Sleepy"
    if arousal == "HIGH" and (eeg_state == "Stressed / Anxious" or hrv_low):
        return "Stressed / Anxious"
    if arousal == "LOW" and eeg_state in ("Relaxed / Calm", "Calm Focus (Flow-like)") and hrv_high:
        return "Calm / Relaxed"
    if eeg_state == "Focused / Concentrating" and arousal in ("MODERATE", "LOW"):
        return "Focused"
    if var_high and arousal in ("MODERATE", "HIGH"):
        return "Emotionally Volatile"
    if eeg_state == "Neutral / Baseline" and arousal == "LOW":
        return "Neutral"
    return f"Mixed ({arousal.lower()} arousal, {eeg_state.lower()})"


def finish_calibration():
    """Build the personal baseline from the RELAX + FOCUS samples."""
    global baseline
    if len(calib_eeg_attention) < 4:
        baseline = {
            "attention_mean": 1.0, "attention_std": 0.5,
            "meditation_mean": 1.0, "meditation_std": 0.5,
            "drowsy_mean": 0.4, "drowsy_std": 0.15,
            "high_beta_mean": 0.15, "high_beta_std": 0.1,
            "gsr_mean": float(np.mean(calib_gsr_means)) if calib_gsr_means else 400.0,
            "gsr_std": float(np.std(calib_gsr_means)) if calib_gsr_means else 100.0,
        }
        return
    baseline = {
        "attention_mean": float(np.mean(calib_eeg_attention)),
        "attention_std": float(np.std(calib_eeg_attention)),
        "meditation_mean": float(np.mean(calib_eeg_meditation)),
        "meditation_std": float(np.std(calib_eeg_meditation)),
        "drowsy_mean": float(np.mean(calib_eeg_drowsy)),
        "drowsy_std": float(np.std(calib_eeg_drowsy)),
        "high_beta_mean": float(np.mean(calib_eeg_high_beta)),
        "high_beta_std": float(np.std(calib_eeg_high_beta)),
        "gsr_mean": float(np.mean(calib_gsr_means)),
        "gsr_std": float(np.std(calib_gsr_means)),
    }


def reset_all():
    """Used by the 'R' key: wipe all data/state (trace + peaks + text)."""
    global mode, calib_phase_start, baseline, samples_since_window
    mode = MODE_IDLE
    calib_phase_start = None
    baseline = None
    samples_since_window = 0

    for buf in (stor1, stor2, stor3):
        buf.clear()
        buf.extend([0] * WINDOW_SIZE)

    sweep_ecg.reset()
    sweep_eeg.reset()
    sweep_gsr.reset()

    calib_eeg_attention.clear()
    calib_eeg_meditation.clear()
    calib_eeg_drowsy.clear()
    calib_eeg_high_beta.clear()
    calib_gsr_means.clear()
    eeg_attention_hist.clear()
    eeg_meditation_hist.clear()

    line_ecg.set_data([], [])
    line_eeg.set_data([], [])
    line_gsr.set_data([], [])
    peaks_marker.set_data([], [])
    ecg_label_text.set_text("HRV, Mean Heart Rate: --")
    eeg_label_text.set_text("State of Emotion: --")
    gsr_label_text.set_text("Arousal State: --")
    calib_status_text.set_text("IDLE")
    time_value_text.set_text("--")
    instructions_text.set_text("Press C to start\ncalibration")
    overall_value_text.set_text("--")
    progress_fill.set_width(0.0)


# ----------------------------------------------------------------------
# Keyboard controls
# ----------------------------------------------------------------------
def on_key(event):
    global mode, calib_phase_start

    key = (event.key or "").lower()

    if key == "c":
        if mode not in (MODE_CALIB_RELAX, MODE_CALIB_FOCUS):
            mode = MODE_CALIB_RELAX
            calib_phase_start = time.time()
            calib_eeg_attention.clear()
            calib_eeg_meditation.clear()
            calib_eeg_drowsy.clear()
            calib_eeg_high_beta.clear()
            calib_gsr_means.clear()

    elif key == "r":
        reset_all()

    elif key == "q":
        try:
            ser.close()
        except Exception:
            pass
        plt.close(fig)


fig.canvas.mpl_connect("key_press_event", on_key)


# ----------------------------------------------------------------------
# Per-window processing (runs every WINDOW_SIZE samples)
# ----------------------------------------------------------------------
def process_window():
    global mode

    # ---- ECG: HR + HRV (computed regardless of mode) ----
    ecg_array = np.array(stor2)
    peaks_ecg, _ = find_peaks(ecg_array, distance=SAMPLE_RATE // 1.2)

    valid_hr = []
    for i in range(len(peaks_ecg) - 1):
        r_length = peaks_ecg[i + 1] - peaks_ecg[i]
        if 92 < r_length < 221:
            valid_hr.append(60 / (r_length / SAMPLE_RATE))
    mean_hr = float(np.mean(valid_hr)) if valid_hr else float("nan")
    hrv = float(np.std(valid_hr)) if valid_hr else None
    ecg_label_text.set_text(
        f"HRV, Mean Heart Rate:  HR={mean_hr:.0f} bpm   HRV={('%.2f' % hrv) if hrv is not None else 'n/a'}"
    )

    # ---- GSR mean (computed regardless of mode) ----
    gsr_mean = float(np.mean(stor1))

    # ---- EEG band powers (computed regardless of mode) ----
    cleaned_eeg = preprocess_eeg_window(list(stor3), SAMPLE_RATE)
    powers = eeg_band_powers(cleaned_eeg, SAMPLE_RATE)
    metrics = eeg_derived_metrics(powers)

    if mode in (MODE_CALIB_RELAX, MODE_CALIB_FOCUS):
        calib_eeg_attention.append(metrics["attention"])
        calib_eeg_meditation.append(metrics["meditation"])
        calib_eeg_drowsy.append(metrics["drowsy"])
        calib_eeg_high_beta.append(metrics["high_beta"])
        calib_gsr_means.append(gsr_mean)
        eeg_label_text.set_text("State of Emotion: (calibrating...)")
        gsr_label_text.set_text("Arousal State: (calibrating...)")
        overall_value_text.set_text("(calibrating...)")

    elif mode == MODE_RUNNING and baseline is not None:
        state, z_att, z_med = classify_eeg_state(metrics, baseline)
        eeg_attention_hist.append(metrics["attention"])
        eeg_meditation_hist.append(metrics["meditation"])
        eeg_hrv = eeg_hrv_index()
        eeg_label_text.set_text(
            f"State of Emotion: {state}  (Att z={z_att:+.2f} Med z={z_med:+.2f}  EEG-HRV="
            f"{('%.3f' % eeg_hrv) if eeg_hrv is not None else 'n/a'})"
        )

        arousal, z_gsr = classify_gsr_arousal(gsr_mean, baseline)
        gsr_label_text.set_text(f"Arousal State: {arousal}  (mean={gsr_mean:.0f}, z={z_gsr:+.2f})")

        emotion = overall_emotion(arousal, hrv, state, eeg_hrv)
        overall_value_text.set_text(emotion)

    else:
        eeg_label_text.set_text("State of Emotion: -- (press S to calibrate)")
        gsr_label_text.set_text(f"Arousal State: -- (raw mean={gsr_mean:.0f})")
        overall_value_text.set_text("-- (press S to calibrate)")


# ----------------------------------------------------------------------
# Animation update
# ----------------------------------------------------------------------
def update(frame):
    global samples_since_window, mode, calib_phase_start

    # --- update sidebar status text + calibration progress bar ---
    if mode == MODE_IDLE:
        calib_status_text.set_text("IDLE")
        time_value_text.set_text("--")
        instructions_text.set_text("Press C to start\ncalibration")
        progress_fill.set_width(0.0)

    elif mode in (MODE_CALIB_RELAX, MODE_CALIB_FOCUS):
        elapsed = time.time() - calib_phase_start
        if mode == MODE_CALIB_RELAX:
            remaining = max(0.0, CALIB_RELAX_SECONDS - elapsed)
            frac = min(1.0, elapsed / CALIB_RELAX_SECONDS)
            calib_status_text.set_text("RELAX PHASE")
            instructions_text.set_text(
                "Sit still, close your\neyes, breathe normally."
            )
            progress_fill.set_facecolor("tab:orange")
            if elapsed >= CALIB_RELAX_SECONDS:
                mode = MODE_CALIB_FOCUS
                calib_phase_start = time.time()
        else:
            remaining = max(0.0, CALIB_FOCUS_SECONDS - elapsed)
            frac = min(1.0, elapsed / CALIB_FOCUS_SECONDS)
            calib_status_text.set_text("FOCUS PHASE")
            instructions_text.set_text(
                "Open eyes, stare at a\nfixed point, concentrate\n"
                "hard (e.g. count back\nfrom 100 by 7s)."
            )
            progress_fill.set_facecolor("tab:purple")
            if elapsed >= CALIB_FOCUS_SECONDS:
                finish_calibration()
                mode = MODE_RUNNING
                frac = 0.0
        time_value_text.set_text(f"{remaining:4.1f}s")
        progress_fill.set_width(0.84 * frac)

    elif mode == MODE_RUNNING:
        calib_status_text.set_text("RUNNING (calibrated)")
        time_value_text.set_text("live")
        instructions_text.set_text("Live monitoring.\nPress R to reset.")
        progress_fill.set_width(0.0)

    # --- read serial data ---
    try:
        while ser.in_waiting > 0:
            line_data = ser.readline().decode("utf-8").strip()
            if not line_data:
                continue
            parts = line_data.split(",")
            if len(parts) != 3:
                continue

            gsr_val = float(parts[0])
            ecg_val = float(parts[1])
            eeg_val = float(parts[2])   # value3 = EEG

            stor1.append(gsr_val)
            stor2.append(ecg_val)
            stor3.append(eeg_val)
            sweep_gsr.push(gsr_val)
            sweep_ecg.push(ecg_val)
            sweep_eeg.push(eeg_val)
            samples_since_window += 1

            if samples_since_window >= WINDOW_SIZE:
                samples_since_window = 0
                process_window()

    except ValueError:
        pass
    except Exception as e:
        overall_value_text.set_text(f"Error: {e}")

    # --- redraw sweep traces from the current buffer state ---
    line_ecg.set_data(SWEEP_X, sweep_ecg.data)
    line_eeg.set_data(SWEEP_X, sweep_eeg.data)
    line_gsr.set_data(SWEEP_X, sweep_gsr.data)

    # --- recompute ECG peak markers fresh from the live buffer, so a
    #     marker vanishes the instant the beam overwrites that sample ---
    y = sweep_ecg.data
    valid_idx = np.where(~np.isnan(y))[0]
    if len(valid_idx) > (SAMPLE_RATE // 1.2) * 2:
        y_valid = y[valid_idx]
        peaks_rel, _ = find_peaks(y_valid, distance=SAMPLE_RATE // 1.2)
        peak_idx = valid_idx[peaks_rel]
        peaks_marker.set_data(peak_idx, y[peak_idx])
    else:
        peaks_marker.set_data([], [])

    return (line_ecg, line_eeg, line_gsr, peaks_marker,
            calib_status_text, time_value_text, instructions_text,
            progress_fill, ecg_label_text, eeg_label_text,
            gsr_label_text, overall_value_text)


ani = animation.FuncAnimation(fig, update, interval=50, blit=False)

plt.show()

try:
    ser.close()
except Exception:
    pass