"""
ECG Signal Quality Detector — Standalone Script
================================================
Run this AFTER ecg_pipeline.py has been trained.

Usage:
    python ecg_detect.py

Set the path to your new CSV file in NEW_PATIENT_FILE below.
"""

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import signal as sp_signal
from scipy.fft import fft, fftfreq
from scipy.stats import skew, kurtosis
from scipy.stats import entropy as sp_entropy

# ─────────────────────────────────────────────────────────────
# CONFIG  ← change these paths
# ─────────────────────────────────────────────────────────────
NEW_PATIENT_FILE = r"C:\Users\darsi\OneDrive\Desktop\majorproject\ecg_project_op\BBB_228.csv"
MODEL_PATH       = r"C:\Users\darsi\OneDrive\Desktop\majorproject\ecg_project_op\ecg_outputs\best_ecg_model.pkl"
OUTPUT_DIR       = r"C:\Users\darsi\OneDrive\Desktop\majorproject\ecg_project_op\ecg_outputs"
FS     = 1000   # sampling frequency Hz
WINDOW = 3000   # samples per window (3 seconds)

PAL = {"clean":"#27AE60", "noisy":"#E74C3C", "accent":"#2C3E50",
       "bg":"#F8F9FA",    "grid":"#DEE2E6",  "blue":"#3498DB"}


# ═════════════════════════════════════════════════════════════
# FEATURE EXTRACTION  (must match ecg_pipeline.py exactly)
# ═════════════════════════════════════════════════════════════
def detect_r_peaks(sig, fs=FS):
    hp_sos   = sp_signal.butter(2, [5, 15], "band", fs=fs, output="sos")
    filtered = sp_signal.sosfiltfilt(hp_sos, sig)
    diff_sq  = np.diff(filtered) ** 2
    win_len  = int(0.15 * fs)
    integrated = np.convolve(diff_sq, np.ones(win_len)/win_len, mode="same")
    thr = 0.5 * np.max(integrated)
    peaks, _ = sp_signal.find_peaks(integrated, height=thr, distance=int(0.35*fs))
    return peaks


def extract_features(ecg, fs=FS):
    f = {}
    n = len(ecg)

    # Time domain
    f["mean"]           = np.mean(ecg)
    f["std"]            = np.std(ecg)
    f["rms"]            = np.sqrt(np.mean(ecg**2))
    f["skewness"]       = float(skew(ecg))
    f["kurtosis"]       = float(kurtosis(ecg))
    f["peak_to_peak"]   = float(np.ptp(ecg))
    f["zero_cross_rate"]= np.sum(np.diff(np.sign(ecg)) != 0) / n
    f["hjorth_mob"]     = np.std(np.diff(ecg)) / (np.std(ecg) + 1e-9)
    f["hjorth_comp"]    = (np.std(np.diff(np.diff(ecg))) /
                           (np.std(np.diff(ecg)) + 1e-9)) / (f["hjorth_mob"] + 1e-9)
    f["flatline_ratio"] = float(np.mean(np.abs(np.diff(ecg)) < 1e-4))

    # Frequency domain
    freq  = fftfreq(n, 1/fs)
    power = np.abs(fft(ecg))**2
    pos   = freq > 0
    fp, pp = freq[pos], power[pos]

    def bp(lo, hi):
        return np.sum(pp[(fp>=lo) & (fp<hi)]) / (np.sum(pp) + 1e-9)

    f["power_vlf"]     = bp(0.003, 0.04)
    f["power_lf"]      = bp(0.04,  0.15)
    f["power_hf"]      = bp(0.15,  0.40)
    f["power_ecg"]     = bp(0.5,   150.0)
    f["power_hfnoise"] = bp(150.0, fs/2)
    f["snr_ratio"]     = f["power_ecg"] / (f["power_hfnoise"] + 1e-9)
    f["spectral_ent"]  = float(sp_entropy(pp/(np.sum(pp)+1e-12)+1e-12))
    f["dom_freq"]      = float(fp[np.argmax(pp)])
    f["pl_50hz"]       = bp(48, 52)
    f["pl_60hz"]       = bp(58, 62)

    # Baseline wander
    lp = sp_signal.sosfiltfilt(
        sp_signal.butter(2, 0.5, "low", fs=fs, output="sos"), ecg)
    f["baseline_wander"] = float(np.std(lp))

    # Morphological / HRV
    r_peaks = detect_r_peaks(ecg, fs)
    if len(r_peaks) >= 2:
        rr = np.diff(r_peaks) / fs * 1000
        f["rr_mean"]    = float(np.mean(rr))
        f["rr_std"]     = float(np.std(rr))
        f["rmssd"]      = float(np.sqrt(np.mean(np.diff(rr)**2)))
        f["pnn50"]      = float(np.mean(np.abs(np.diff(rr)) > 50))
        f["hr_bpm"]     = 60_000 / (np.mean(rr) + 1e-9)
        r_amps          = ecg[r_peaks]
        f["r_amp_mean"] = float(np.mean(r_amps))
        f["r_amp_std"]  = float(np.std(r_amps))
        f["r_amp_cv"]   = f["r_amp_std"] / (abs(f["r_amp_mean"]) + 1e-9)
        f["n_beats"]    = len(r_peaks)
        f["beat_reg"]   = 1 - (f["rr_std"] / (f["rr_mean"] + 1e-9))
    else:
        for k in ["rr_mean","rr_std","rmssd","pnn50","hr_bpm",
                  "r_amp_mean","r_amp_std","r_amp_cv","n_beats","beat_reg"]:
            f[k] = 0.0

    return np.array(list(f.values()), dtype=np.float32), list(f.keys())


def segment_windows(signal, window=WINDOW):
    n = len(signal) // window
    return [signal[i*window:(i+1)*window] for i in range(n)]


# ═════════════════════════════════════════════════════════════
# LOAD CSV
# ═════════════════════════════════════════════════════════════
def load_csv(path):
    df = pd.read_csv(path, header=0, skiprows=[1])
    df.columns = [str(c).strip().strip("'").strip().lower() for c in df.columns]
    if "ii" not in df.columns:
        print(f"  Available columns: {list(df.columns)}")
        raise ValueError("Column 'ii' not found. Check column names above.")
    sig = df["ii"].to_numpy(dtype=np.float64)
    sig = sig[~np.isnan(sig)]
    return sig


# ═════════════════════════════════════════════════════════════
# DETECTION IMAGE
# ═════════════════════════════════════════════════════════════
def save_detection_image(windows, preds, confs, fname, save_path):
    n     = min(8, len(windows))
    ncols = 2
    nrows = (n + 1) // 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, nrows*3.5), facecolor="white")
    fig.suptitle(f"ECG Detection Report — {fname}",
                 fontsize=13, fontweight="bold", y=0.99)
    t3 = np.arange(WINDOW) / FS

    ax_list = axes.flat if hasattr(axes, "flat") else [axes]
    for idx in range(n):
        ax    = ax_list[idx] if n > 1 else axes
        win   = windows[idx]
        pred  = preds[idx]
        conf  = confs[idx]
        sc    = PAL["clean"] if pred == 0 else PAL["noisy"]
        label = "CLEAN ✓" if pred == 0 else "NOISY  ✗"
        t_start = idx * WINDOW / FS

        ax.set_facecolor(PAL["bg"])
        ax.plot(t3, win, color=sc, lw=0.7, alpha=0.9)
        ax.grid(True, color=PAL["grid"], lw=0.4, ls="--", alpha=0.6)
        ax.spines[["top","right"]].set_visible(False)
        ax.set_xlim(0, WINDOW/FS)
        ax.set_xlabel("Time (s)", fontsize=7)
        ax.set_ylabel("mV", fontsize=7)
        ax.tick_params(labelsize=6)
        ax.set_title(
            f"Window {idx+1}  [{t_start:.1f}s – {t_start+3:.1f}s]  "
            f"→  {label}  ({conf*100:.1f}%)",
            fontsize=8.5, fontweight="bold", color=sc, pad=4)

        # Confidence bar at bottom
        y_lo, y_hi = ax.get_ylim()
        strip = (y_hi - y_lo) * 0.06
        ax.fill_between(
            np.linspace(0, WINDOW/FS * conf, 50),
            y_lo, y_lo + strip,
            color=sc, alpha=0.35, zorder=3)

    # Hide unused subplots
    total_axes = nrows * ncols
    for idx in range(n, total_axes):
        ax_list[idx].set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Image saved   → {save_path}")


# ═════════════════════════════════════════════════════════════
# SUMMARY IMAGE  (full signal with colour-coded windows)
# ═════════════════════════════════════════════════════════════
def save_summary_image(signal, preds, fname, save_path):
    fig, ax = plt.subplots(figsize=(20, 4), facecolor="white")
    t = np.arange(len(signal)) / FS
    ax.plot(t, signal, color="#BDC3C7", lw=0.5, alpha=0.6, zorder=1)

    for i, pred in enumerate(preds):
        start = i * WINDOW
        end   = start + WINDOW
        color = PAL["clean"] if pred == 0 else PAL["noisy"]
        ax.axvspan(start/FS, end/FS, alpha=0.25, color=color, zorder=2)
        label = "C" if pred == 0 else "N"
        ax.text((start + WINDOW/2) / FS,
                ax.get_ylim()[1] * 0.85 if ax.get_ylim()[1] != 1.0 else 0.8,
                label, ha="center", fontsize=7, fontweight="bold", color=color)

    ax.set_facecolor(PAL["bg"])
    ax.grid(True, color=PAL["grid"], lw=0.4, ls="--", alpha=0.6)
    ax.spines[["top","right"]].set_visible(False)
    ax.set_xlabel("Time (s)", fontsize=9)
    ax.set_ylabel("Amplitude (mV)", fontsize=9)
    ax.set_title(f"Full Signal Quality Map — {fname}   "
                 f"[Green = CLEAN  |  Red = NOISY]",
                 fontsize=11, fontweight="bold")

    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=PAL["clean"], alpha=0.5, label="CLEAN"),
                       Patch(color=PAL["noisy"], alpha=0.5, label="NOISY")],
              loc="upper right", fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Signal map    → {save_path}")


# ═════════════════════════════════════════════════════════════
# MAIN DETECTION
# ═════════════════════════════════════════════════════════════
def main():
    print("\n" + "═"*60)
    print("  ECG Signal Quality Detector")
    print("═"*60)

    # Check files exist
    if not os.path.exists(NEW_PATIENT_FILE):
        print(f"\n  [ERROR] File not found: {NEW_PATIENT_FILE}")
        print("  → Set NEW_PATIENT_FILE at the top of this script.")
        return

    if not os.path.exists(MODEL_PATH):
        print(f"\n  [ERROR] Model not found: {MODEL_PATH}")
        print("  → Run ecg_pipeline.py first to train and save the model.")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    fname = os.path.basename(NEW_PATIENT_FILE)

    # Load model
    print(f"\n  Loading model  → {MODEL_PATH}")
    with open(MODEL_PATH, "rb") as f:
        saved = pickle.load(f)
    model = saved["model"]
    print(f"  Best model     : {saved.get('best_name', 'Unknown')}")

    # Load signal
    print(f"\n  Loading signal → {NEW_PATIENT_FILE}")
    signal  = load_csv(NEW_PATIENT_FILE)
    print(f"  Samples loaded : {len(signal):,}  ({len(signal)/FS:.1f} seconds)")

    # Segment
    windows = segment_windows(signal, WINDOW)
    print(f"  Windows (3s)   : {len(windows)}\n")

    # Predict
    preds, confs = [], []
    print(f"  {'Window':>6}  {'Time Range':>16}  {'Result':>10}  {'Confidence':>12}")
    print(f"  {'─'*6}  {'─'*16}  {'─'*10}  {'─'*12}")

    for i, win in enumerate(windows):
        feats, _ = extract_features(win, FS)
        pred     = model.predict([feats])[0]
        proba    = model.predict_proba([feats])
        conf     = (proba[:, 1] if proba.shape[1] > 1 else proba[:, 0])[0]
        preds.append(pred)
        confs.append(conf)
        t_start  = i * WINDOW / FS
        t_end    = t_start + WINDOW / FS
        label    = "CLEAN ✓" if pred == 0 else "NOISY  ✗"
        print(f"  {i+1:>6}  {t_start:6.1f}s – {t_end:5.1f}s  {label:>10}  {conf*100:>10.1f}%")

    # Summary stats
    n_clean = sum(p == 0 for p in preds)
    n_noisy = sum(p == 1 for p in preds)
    print(f"\n  {'─'*50}")
    print(f"  Total windows : {len(preds)}")
    print(f"  CLEAN         : {n_clean} ({100*n_clean/len(preds):.1f}%)")
    print(f"  NOISY         : {n_noisy} ({100*n_noisy/len(preds):.1f}%)")
    print(f"  {'─'*50}")

    # Save results CSV
    base     = os.path.splitext(fname)[0]
    csv_out  = os.path.join(OUTPUT_DIR, f"detection_{base}.csv")
    rows     = [(i+1, i*WINDOW/FS, (i+1)*WINDOW/FS,
                 "CLEAN" if p==0 else "NOISY", round(c*100, 1))
                for i, (p, c) in enumerate(zip(preds, confs))]
    pd.DataFrame(rows, columns=["Window","Start_s","End_s",
                                 "Label","Confidence_%"]).to_csv(csv_out, index=False)
    print(f"\n  Results CSV   → {csv_out}")

    # Save images
    img1 = os.path.join(OUTPUT_DIR, f"detection_{base}_windows.png")
    img2 = os.path.join(OUTPUT_DIR, f"detection_{base}_map.png")
    save_detection_image(windows, preds, confs, fname, img1)
    save_summary_image(signal, preds, fname, img2)

    print("\n" + "═"*60)
    print("  ✓  Detection complete!")
    print("═"*60 + "\n")


if __name__ == "__main__":
    main()
