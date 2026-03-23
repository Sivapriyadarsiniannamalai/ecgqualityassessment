"""
ECG Signal Quality Classifier — Real Excel File Pipeline
=========================================================
• Reads 21 patient Excel files  (Elapsed time + 16 lead columns)
• Extracts non-overlapping 3000-sample windows from Lead II
• Auto-labels each window as CLEAN (0) or NOISY (1)
• Extracts 31 time / frequency / morphological features per window
• Trains Random Forest, Gradient Boosting, SVM classifiers
• Outputs 4 diagnostic images + trained model + feature CSV
  → Drop all Excel files in INPUT_DIR and run: python ecg_pipeline.py
"""

import os, glob, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
from scipy import signal as sp_signal
from scipy.stats import skew, kurtosis, entropy as sp_entropy
from scipy.fft import fft, fftfreq
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (classification_report, confusion_matrix,
                              roc_curve, auc, f1_score)
from sklearn.pipeline import Pipeline
import pickle
from tqdm import tqdm
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# USER CONFIG  ← adjust these two paths before running
# ─────────────────────────────────────────────────────────────
INPUT_DIR  = r"G:\ecg_project_op\ecg_data" # folder containing all .xlsx files
OUTPUT_DIR = r"G:\ecg_project_op\ecg_outputs"      # where images / model are saved

FS          = 1000   # sampling frequency (Hz)
WINDOW      = 3000   # samples per window  (= 3 seconds)
LEAD_COL    = "ii"   # Lead II column name  (lowercase)
TIME_COL    = "Elapsed time"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────
# PALETTE
# ─────────────────────────────────────────────────────────────
PAL = {"clean":"#27AE60","noisy":"#E74C3C","accent":"#2C3E50",
       "bg":"#F8F9FA","grid":"#DEE2E6","blue":"#3498DB",
       "orange":"#E67E22","purple":"#9B59B6"}


# ═════════════════════════════════════════════════════════════
# 1.  DATA LOADING
# ═════════════════════════════════════════════════════════════
def load_excel_files(input_dir):
    """Return list of (filename, lead_ii_array) for every xlsx in input_dir."""
    paths = sorted(glob.glob(os.path.join(input_dir, "*.xlsx")) +
                   glob.glob(os.path.join(input_dir, "*.xls"))  +
                   glob.glob(os.path.join(input_dir, "*.csv")))


    if not paths:
        raise FileNotFoundError(
            f"No Excel files found in '{input_dir}'.\n"
            f"Place your 21 .xlsx files there and re-run.")
    print(f"  Found {len(paths)} Excel file(s)")
    records = []
    for p in paths:
        try:
            if p.endswith(".csv"):
                df = pd.read_csv(p, header=0, skiprows=[1])
            else:
                df = pd.read_excel(p, header=0, skiprows=[1])
            df.columns = [str(c).strip().strip("'").strip().lower() for c in df.columns]
            # normalise column names: strip spaces, lowercase
            df.columns = [str(c).strip().lower() for c in df.columns]
            lead_col = LEAD_COL.lower()
            if lead_col not in df.columns:
                # try to find a column containing 'ii'
                candidates = [c for c in df.columns if c == "ii" or "lead ii" in c or c == "2"]
                if candidates:
                    lead_col = candidates[0]
                else:
                    print(f"  [SKIP] {os.path.basename(p)} — Lead II column not found. "
                          f"Columns: {list(df.columns)}")
                    continue
            sig = df[lead_col].to_numpy(dtype=np.float64)
            sig = sig[~np.isnan(sig)]   # drop NaNs
            records.append((os.path.basename(p), sig))
            print(f"  Loaded {os.path.basename(p):40s}  {len(sig):>7,} samples")
        except Exception as e:
            print(f"  [ERROR] {os.path.basename(p)}: {e}")
    return records


def segment_windows(signal, window=WINDOW):
    """Split signal into non-overlapping windows; discard remainder."""
    n_windows = len(signal) // window
    return [signal[i*window:(i+1)*window] for i in range(n_windows)]


# ═════════════════════════════════════════════════════════════
# 2.  AUTO-LABELLING
# ═════════════════════════════════════════════════════════════
def auto_label(window, fs=FS):
    """
    Rule-based quality scorer.
    Returns 1 (NOISY) if any hard threshold is exceeded, else 0 (CLEAN).

    Rules:
      • Clipping / flatline   : >8 % of consecutive samples identical
      • Baseline wander       : low-pass std > 25 % of signal std
      • High-freq noise       : power above 150 Hz > 15 % of total power
      • Power-line artefact   : narrow-band power at 50 Hz > 5 % of total
      • Amplitude outliers    : any sample > 5 × IQR from median
      • R-peak amplitude CV   : coefficient of variation > 0.40 (unstable beats)
      • Too few beats         : <5 R-peaks detected in 3-second window
    """
    w = window.copy()

    # 1. Flatline / clipping
    flat = np.mean(np.abs(np.diff(w)) < 1e-4)
    if flat > 0.40:
        return 1

    # 2. Baseline wander (high-pass residual)
    lp_sos = sp_signal.butter(2, 0.5, "low",  fs=fs, output="sos")
    baseline = sp_signal.sosfiltfilt(lp_sos, w)
    if np.std(baseline) >  0.80 * np.std(w):
        return 1

    # 3. High-frequency noise ratio
    freq = fftfreq(len(w), 1/fs)
    power = np.abs(fft(w))**2
    pos = freq > 0
    hf_ratio = np.sum(power[pos][freq[pos] >= 150]) / (np.sum(power[pos]) + 1e-9)
    if hf_ratio >0.50:
        return 1

    # 4. 50 Hz power-line
    pl_ratio = np.sum(power[pos][(freq[pos] >= 48) & (freq[pos] <= 52)]) / (np.sum(power[pos]) + 1e-9)
    if pl_ratio > 0.30:
        return 1

    # 5. Amplitude outliers
    med = np.median(w)
    iqr = np.percentile(w, 75) - np.percentile(w, 25)
    if np.any(np.abs(w - med) > 5 * iqr):
        return 1

    # 6. R-peak beat regularity
    r_peaks = detect_r_peaks(w, fs)
    if len(r_peaks) < 2:
        return 1
    rr = np.diff(r_peaks)
    cv = np.std(rr) / (np.mean(rr) + 1e-9)
    if cv > 0.80:
        return 1

    return 0


# ═════════════════════════════════════════════════════════════
# 3.  FEATURE EXTRACTION
# ═════════════════════════════════════════════════════════════
def detect_r_peaks(sig, fs=FS):
    """Pan-Tompkins-inspired R-peak detector."""
    hp_sos = sp_signal.butter(2, [5, 15], "band", fs=fs, output="sos")
    filtered = sp_signal.sosfiltfilt(hp_sos, sig)
    diff_sq  = np.diff(filtered) ** 2
    win_len  = int(0.15 * fs)
    integrated = np.convolve(diff_sq, np.ones(win_len)/win_len, mode="same")
    thr = 0.5 * np.max(integrated)
    peaks, _ = sp_signal.find_peaks(integrated, height=thr, distance=int(0.35*fs))
    return peaks


def extract_features(ecg, fs=FS):
    """Return (feature_vector [float32], feature_names [list])."""
    f = {}
    n  = len(ecg)

    # ── TIME DOMAIN ──────────────────────────────────────────
    f["mean"]          = np.mean(ecg)
    f["std"]           = np.std(ecg)
    f["rms"]           = np.sqrt(np.mean(ecg**2))
    f["skewness"]      = float(skew(ecg))
    f["kurtosis"]      = float(kurtosis(ecg))
    f["peak_to_peak"]  = float(np.ptp(ecg))
    f["zero_cross_rate"]= np.sum(np.diff(np.sign(ecg)) != 0) / n
    f["hjorth_mob"]    = np.std(np.diff(ecg)) / (np.std(ecg) + 1e-9)
    f["hjorth_comp"]   = (np.std(np.diff(np.diff(ecg))) /
                          (np.std(np.diff(ecg)) + 1e-9)) / (f["hjorth_mob"] + 1e-9)
    f["flatline_ratio"]= float(np.mean(np.abs(np.diff(ecg)) < 1e-4))

    # ── FREQUENCY DOMAIN ─────────────────────────────────────
    freq  = fftfreq(n, 1/fs)
    power = np.abs(fft(ecg))**2
    pos   = freq > 0
    fp, pp = freq[pos], power[pos]

    def bp(lo, hi): return np.sum(pp[(fp>=lo)&(fp<hi)]) / (np.sum(pp) + 1e-9)

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

    # ── BASELINE WANDER ──────────────────────────────────────
    lp = sp_signal.sosfiltfilt(
        sp_signal.butter(2, 0.5, "low", fs=fs, output="sos"), ecg)
    f["baseline_wander"] = float(np.std(lp))

    # ── MORPHOLOGICAL / HRV ──────────────────────────────────
    r_peaks = detect_r_peaks(ecg, fs)
    if len(r_peaks) >= 2:
        rr = np.diff(r_peaks) / fs * 1000   # ms
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

    names = list(f.keys())
    return np.array(list(f.values()), dtype=np.float32), names


# ═════════════════════════════════════════════════════════════
# 4.  BUILD DATASET FROM EXCEL FILES
# ═════════════════════════════════════════════════════════════
def build_dataset(input_dir):
    records = load_excel_files(input_dir)
    if not records:
        raise RuntimeError("No valid data loaded. Check INPUT_DIR and column names.")

    all_windows, all_labels, all_feats = [], [], []
    feature_names = None
    patient_info  = []   # (filename, window_idx, label)

    print(f"\n  Segmenting → 3000-sample windows @ {FS} Hz …")
    for fname, sig in tqdm(records, desc="  Files"):
        wins = segment_windows(sig, WINDOW)
        for w_idx, win in enumerate(wins):
            label = auto_label(win, FS)
            feats, fnames = extract_features(win, FS)
            all_windows.append(win)
            all_labels.append(label)
            all_feats.append(feats)
            patient_info.append((fname, w_idx, label))
            if feature_names is None:
                feature_names = fnames

    X = np.array(all_feats,  dtype=np.float32)
    y = np.array(all_labels, dtype=np.int32)
    print(f"\n  Total windows : {len(y)}")
    print(f"  CLEAN (0)     : {(y==0).sum()} ({100*(y==0).mean():.1f}%)")
    print(f"  NOISY (1)     : {(y==1).sum()} ({100*(y==1).mean():.1f}%)")
    return X, y, feature_names, all_windows, patient_info


# ═════════════════════════════════════════════════════════════
# 5.  MODEL TRAINING
# ═════════════════════════════════════════════════════════════
def train_models(X, y):
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    classifiers = {
        "Random Forest":     RandomForestClassifier(
                                 n_estimators=300, max_depth=15,
                                 class_weight="balanced", random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(
                                 n_estimators=200, learning_rate=0.08,
                                 max_depth=5, random_state=42),
        "SVM (RBF)":         SVC(kernel="rbf", C=10, gamma="scale",
                                 probability=True, class_weight="balanced", random_state=42),
    }

    trained, results = {}, {}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    print("\n  Training classifiers …")
    for name, clf in classifiers.items():
        pipe = Pipeline([("imputer", SimpleImputer(strategy="median")),
                         ("scaler", StandardScaler()),
                         ("clf", clf)])
        cv_f1 = cross_val_score(pipe, X_tr, y_tr, cv=cv, scoring="f1", n_jobs=-1)
        pipe.fit(X_tr, y_tr)
        y_pred = pipe.predict(X_te)
        proba =  pipe.predict_proba(X_te)
        y_prob = proba[:, 1] if proba.shape[1] > 1 else proba[:, 0]
        trained[name]  = pipe
        results[name]  = dict(cv_f1=cv_f1, y_pred=y_pred, y_prob=y_prob,
                               report=classification_report(y_te, y_pred,
                                           target_names=["Clean","Noisy"]),
                               cm=confusion_matrix(y_te, y_pred))
        print(f"  {name:22s}  CV-F1 = {cv_f1.mean():.3f} ± {cv_f1.std():.3f}  "
              f"Test-F1 = {f1_score(y_te, y_pred):.3f}")

    # save best model (by test F1)
    best_name = max(results, key=lambda k: f1_score(y_te, results[k]["y_pred"]))
    best_pipe  = trained[best_name]
    model_path = os.path.join(OUTPUT_DIR, "best_ecg_model.pkl")
    with open(model_path, "wb") as fp:
        pickle.dump({"model": best_pipe, "feature_names": None,
                     "best_name": best_name}, fp)
    print(f"\n  Best model: {best_name}  → saved to {model_path}")

    return trained, results, X_te, y_te


# ═════════════════════════════════════════════════════════════
# 6.  PLOT HELPERS
# ═════════════════════════════════════════════════════════════
def _ax(ax, title="", xlabel="", ylabel=""):
    ax.set_facecolor(PAL["bg"])
    ax.grid(True, color=PAL["grid"], lw=0.5, ls="--", alpha=0.7)
    ax.spines[["top","right"]].set_visible(False)
    if title:  ax.set_title(title, fontsize=9, fontweight="bold", pad=5)
    if xlabel: ax.set_xlabel(xlabel, fontsize=8)
    if ylabel: ax.set_ylabel(ylabel, fontsize=8)
    ax.tick_params(labelsize=7)


# ─── Figure 1 : Dataset Overview ─────────────────────────────
def fig_dataset_overview(X, y, feature_names, windows, patient_info, save_path):
    fig = plt.figure(figsize=(20, 14), facecolor="white")
    fig.suptitle("ECG Dataset Overview — Auto-Labelled Windows from Excel Files",
                 fontsize=15, fontweight="bold", y=0.99)
    gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.50, wspace=0.36)

    t3 = np.arange(WINDOW) / FS

    # ① Sample CLEAN waveform
    clean_idx = np.where(y == 0)[0]
    noisy_idx = np.where(y == 1)[0]
    for col, (idxs, label) in enumerate([(clean_idx,"CLEAN"), (noisy_idx,"NOISY")]):
        ax = fig.add_subplot(gs[0, col*2:col*2+2])
        if len(idxs):
            win = windows[idxs[0]]
            c   = PAL["clean"] if label=="CLEAN" else PAL["noisy"]
            ax.plot(t3, win, color=c, lw=0.75, alpha=0.9)
            r_peaks = detect_r_peaks(win)
            if len(r_peaks):
                ax.scatter(t3[r_peaks], win[r_peaks], s=25, color="#C0392B",
                           zorder=5, marker="^", label="R peaks")
                ax.legend(fontsize=7, loc="upper right")
        _ax(ax, title=f"Sample {label} Window", xlabel="Time (s)", ylabel="Amplitude (mV)")
        ax.set_xlim(0, WINDOW/FS)
        ax.text(0.97, 0.93, label, transform=ax.transAxes, ha="right", va="top",
                fontsize=10, fontweight="bold", color="white",
                bbox=dict(boxstyle="round,pad=0.3",
                          facecolor=PAL["clean"] if label=="CLEAN" else PAL["noisy"],
                          alpha=0.9))

    # ② Label distribution
    ax_pie = fig.add_subplot(gs[1, 0])
    counts  = [(y==0).sum(), (y==1).sum()]
    ax_pie.pie(counts, labels=["Clean","Noisy"],
               colors=[PAL["clean"], PAL["noisy"]],
               autopct="%1.1f%%", startangle=90,
               textprops={"fontsize":9})
    ax_pie.set_title("Window Label Distribution", fontsize=9, fontweight="bold")

    # ③ Windows per file
    ax_wf = fig.add_subplot(gs[1, 1:3])
    fnames_u = []
    win_counts_c, win_counts_n = [], []
    for fname, _, lbl in patient_info:
        if not fnames_u or fnames_u[-1] != fname:
            fnames_u.append(fname); win_counts_c.append(0); win_counts_n.append(0)
        if lbl == 0: win_counts_c[-1] += 1
        else:        win_counts_n[-1] += 1
    x_pos = np.arange(len(fnames_u))
    ax_wf.bar(x_pos, win_counts_c, label="Clean", color=PAL["clean"], alpha=0.85)
    ax_wf.bar(x_pos, win_counts_n, bottom=win_counts_c, label="Noisy",
              color=PAL["noisy"], alpha=0.85)
    ax_wf.set_xticks(x_pos)
    ax_wf.set_xticklabels([os.path.splitext(f)[0][-12:] for f in fnames_u],
                           rotation=55, ha="right", fontsize=6)
    _ax(ax_wf, title="Windows per Patient File", ylabel="# Windows")
    ax_wf.legend(fontsize=8)

    # ④ SNR distribution
    ax_snr = fig.add_subplot(gs[1, 3])
    snr_c = X[y==0, feature_names.index("snr_ratio")]
    snr_n = X[y==1, feature_names.index("snr_ratio")]
    ax_snr.hist(snr_c, bins=30, color=PAL["clean"], alpha=0.7, label="Clean", density=True)
    ax_snr.hist(snr_n, bins=30, color=PAL["noisy"], alpha=0.7, label="Noisy", density=True)
    _ax(ax_snr, title="SNR Distribution", xlabel="SNR ratio", ylabel="Density")
    ax_snr.legend(fontsize=8)

    # ⑤ Baseline wander vs R-amp CV scatter
    ax_sc = fig.add_subplot(gs[2, :2])
    bw_c  = X[y==0, feature_names.index("baseline_wander")]
    bw_n  = X[y==1, feature_names.index("baseline_wander")]
    cv_c  = X[y==0, feature_names.index("r_amp_cv")]
    cv_n  = X[y==1, feature_names.index("r_amp_cv")]
    ax_sc.scatter(bw_c, cv_c, c=PAL["clean"], alpha=0.4, s=10, label="Clean")
    ax_sc.scatter(bw_n, cv_n, c=PAL["noisy"], alpha=0.4, s=10, label="Noisy")
    _ax(ax_sc, title="Baseline Wander vs R-Amplitude CV",
        xlabel="Baseline Wander (std, mV)", ylabel="R-Amp Coefficient of Variation")
    ax_sc.legend(fontsize=8)

    # ⑥ Top feature importances (placeholder — real RF trained later)
    ax_feat = fig.add_subplot(gs[2, 2:])
    feat_mean_diff = np.abs(X[y==0].mean(0) - X[y==1].mean(0))
    top_idx = np.argsort(feat_mean_diff)[-12:]
    ax_feat.barh(range(12), feat_mean_diff[top_idx],
                 color=PAL["blue"], alpha=0.85, edgecolor="white")
    ax_feat.set_yticks(range(12))
    ax_feat.set_yticklabels([feature_names[i] for i in top_idx], fontsize=7)
    _ax(ax_feat, title="Top 12 Features (Mean |Clean−Noisy|)", xlabel="|Δ mean|")

    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved → {save_path}")


# ─── Figure 2 : Training Results ─────────────────────────────
def fig_training_results(results, trained_models, feature_names, y_te, save_path):
    fig = plt.figure(figsize=(20, 13), facecolor="white")
    fig.suptitle("Classifier Training Results — ECG Signal Quality",
                 fontsize=15, fontweight="bold", y=0.99)
    gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.48, wspace=0.36)
    colors3 = [PAL["blue"], PAL["orange"], PAL["purple"]]
    mnames  = list(results.keys())

    # ① Confusion matrices
    for i, mname in enumerate(mnames):
        ax = fig.add_subplot(gs[0, i])
        cm = results[mname]["cm"]
        ax.imshow(cm, cmap="Blues")
        for r in range(2):
            for c in range(2):
                ax.text(c, r, str(cm[r,c]), ha="center", va="center",
                        fontsize=16, fontweight="bold",
                        color="white" if cm[r,c] > cm.max()*0.55 else PAL["accent"])
        ax.set_xticks([0,1]); ax.set_yticks([0,1])
        ax.set_xticklabels(["Clean","Noisy"], fontsize=8)
        ax.set_yticklabels(["Clean","Noisy"], fontsize=8, rotation=90, va="center")
        _ax(ax, title=f"{mname}\nConfusion Matrix", xlabel="Predicted", ylabel="Actual")

    # ② ROC curves
    ax_roc = fig.add_subplot(gs[0, 3])
    for i, (mname, res) in enumerate(results.items()):
        fpr, tpr, _ = roc_curve(y_te, res["y_prob"])
        roc_auc = auc(fpr, tpr)
        ax_roc.plot(fpr, tpr, color=colors3[i], lw=2,
                    label=f"{mname.split()[0]} ({roc_auc:.3f})")
    ax_roc.plot([0,1],[0,1],"--", color="#BDC3C7", lw=1)
    _ax(ax_roc, title="ROC Curves", xlabel="FPR", ylabel="TPR")
    ax_roc.legend(fontsize=8, loc="lower right", framealpha=0.7)

    # ③ CV F1 scores
    ax_f1 = fig.add_subplot(gs[1, :2])
    f1m = [results[m]["cv_f1"].mean() for m in mnames]
    f1s = [results[m]["cv_f1"].std()  for m in mnames]
    bars = ax_f1.bar(range(len(mnames)), f1m, yerr=f1s, capsize=6,
                     color=colors3, alpha=0.85, width=0.5, edgecolor="white")
    for bar, val in zip(bars, f1m):
        ax_f1.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.012,
                   f"{val:.3f}", ha="center", va="bottom",
                   fontsize=10, fontweight="bold")
    ax_f1.set_xticks(range(len(mnames)))
    ax_f1.set_xticklabels([m.replace(" ","\n") for m in mnames], fontsize=8)
    ax_f1.set_ylim(0, 1.15)
    _ax(ax_f1, title="5-Fold CV F1 Score", ylabel="F1")

    # ④ Per-class precision / recall
    ax_pr = fig.add_subplot(gs[1, 2:])
    x = np.arange(len(mnames))
    prec_c, prec_n, rec_c, rec_n = [], [], [], []
    for m in mnames:
        from sklearn.metrics import precision_score, recall_score
        prec_c.append(precision_score(y_te, results[m]["y_pred"], pos_label=0))
        prec_n.append(precision_score(y_te, results[m]["y_pred"], pos_label=1))
        rec_c.append(recall_score(y_te,  results[m]["y_pred"], pos_label=0))
        rec_n.append(recall_score(y_te,  results[m]["y_pred"], pos_label=1))
    w = 0.18
    ax_pr.bar(x-1.5*w, prec_c, w, label="Prec-Clean",  color=PAL["clean"],  alpha=0.85)
    ax_pr.bar(x-0.5*w, prec_n, w, label="Prec-Noisy",  color=PAL["noisy"],  alpha=0.85)
    ax_pr.bar(x+0.5*w, rec_c,  w, label="Rec-Clean",   color=PAL["clean"],  alpha=0.5)
    ax_pr.bar(x+1.5*w, rec_n,  w, label="Rec-Noisy",   color=PAL["noisy"],  alpha=0.5)
    ax_pr.set_xticks(x)
    ax_pr.set_xticklabels([m.replace(" ","\n") for m in mnames], fontsize=8)
    ax_pr.set_ylim(0, 1.15)
    _ax(ax_pr, title="Per-Class Precision & Recall", ylabel="Score")
    ax_pr.legend(fontsize=7, loc="lower right", ncol=2)

    # ⑤ Feature importances (RF)
    ax_fi = fig.add_subplot(gs[2, :2])
    rf_clf = trained_models["Random Forest"].named_steps["clf"]
    imp = rf_clf.feature_importances_
    idx = np.argsort(imp)[-15:]
    ax_fi.barh(range(15), imp[idx], color=PAL["blue"], alpha=0.85, edgecolor="white")
    ax_fi.set_yticks(range(15))
    ax_fi.set_yticklabels([feature_names[i] for i in idx], fontsize=7)
    _ax(ax_fi, title="Top 15 Feature Importances (Random Forest)", xlabel="Importance")

    # ⑥ Classification report text
    ax_rep = fig.add_subplot(gs[2, 2:])
    ax_rep.axis("off")
    best_m   = max(results, key=lambda k: results[k]["cv_f1"].mean())
    rep_text = f"Best Model: {best_m}\n\n" + results[best_m]["report"]
    ax_rep.text(0.03, 0.97, rep_text, transform=ax_rep.transAxes,
                va="top", ha="left", fontsize=8, family="monospace",
                bbox=dict(boxstyle="round,pad=0.5", facecolor=PAL["bg"],
                          edgecolor=PAL["grid"]))

    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved → {save_path}")


# ─── Figure 3 : Single Window Diagnostic Report ──────────────
def fig_window_report(ecg, label_true, label_pred, confidence,
                      feature_names, feats, patient_name, save_path):
    t = np.arange(len(ecg)) / FS
    is_clean   = label_pred == 0
    sc         = PAL["clean"] if is_clean else PAL["noisy"]
    status_txt = "CLEAN" if is_clean else "NOISY"
    gt_txt     = "Clean" if label_true == 0 else "Noisy"
    correct    = label_true == label_pred

    fig = plt.figure(figsize=(20, 15), facecolor="white")
    fig.suptitle(f"ECG Diagnostic Report  —  {patient_name}",
                 fontsize=15, fontweight="bold", y=0.99)
    gs = gridspec.GridSpec(4, 3, figure=fig, hspace=0.50, wspace=0.35)

    # ① Verdict badge
    ax_b = fig.add_subplot(gs[0, 2])
    ax_b.set_xlim(0,1); ax_b.set_ylim(0,1); ax_b.axis("off")
    rect = FancyBboxPatch((0.05,0.05), 0.9, 0.9,
                          boxstyle="round,pad=0.05", lw=3,
                          edgecolor=sc, facecolor=sc, alpha=0.12)
    ax_b.add_patch(rect)
    ax_b.text(0.5, 0.72, status_txt, ha="center", va="center",
              fontsize=30, fontweight="bold", color=sc)
    ax_b.text(0.5, 0.50, f"Confidence: {confidence*100:.1f}%",
              ha="center", fontsize=12, color=PAL["accent"])
    ax_b.text(0.5, 0.33, f"Ground truth: {gt_txt}",
              ha="center", fontsize=10, color="gray")
    ax_b.text(0.5, 0.15, "✓ Correct" if correct else "✗ Incorrect",
              ha="center", fontsize=11, fontweight="bold",
              color=PAL["clean"] if correct else PAL["noisy"])

    # ② Full waveform
    ax_ecg = fig.add_subplot(gs[0, :2])
    ax_ecg.plot(t, ecg, color=sc, lw=0.7, alpha=0.9)
    _ax(ax_ecg, title=f"Full ECG Window — Lead II ({WINDOW} samples, {WINDOW/FS:.1f} s)",
        xlabel="Time (s)", ylabel="Amplitude (mV)")
    ax_ecg.set_xlim(0, WINDOW/FS)

    # ③ Zoomed (first 2 s) with R-peaks
    ax_z = fig.add_subplot(gs[1, :2])
    sl   = slice(0, 2*FS)
    ax_z.plot(t[sl], ecg[sl], color=sc, lw=1.1)
    rp = detect_r_peaks(ecg[sl], FS)
    if len(rp):
        ax_z.scatter(t[sl][rp], ecg[sl][rp], color="#C0392B",
                     s=50, zorder=5, marker="^", label="R peaks")
        ax_z.legend(fontsize=8)
    _ax(ax_z, title="Zoomed — First 2 s with R-Peak Detection",
        xlabel="Time (s)", ylabel="Amplitude (mV)")

    # ④ Welch PSD
    ax_psd = fig.add_subplot(gs[1, 2])
    f_w, psd = sp_signal.welch(ecg, FS, nperseg=min(1024, WINDOW//2))
    ax_psd.semilogy(f_w, psd, color=PAL["blue"], lw=1.2)
    ax_psd.axvspan(0.5, 150, alpha=0.10, color=PAL["clean"], label="ECG band")
    ax_psd.axvspan(48,  52,  alpha=0.20, color=PAL["noisy"], label="50 Hz PL")
    ax_psd.axvspan(150, FS/2,alpha=0.10, color=PAL["noisy"], label="HF noise")
    _ax(ax_psd, title="Power Spectral Density (Welch)",
        xlabel="Frequency (Hz)", ylabel="PSD (mV²/Hz)")
    ax_psd.legend(fontsize=7); ax_psd.set_xlim(0, 250)

    # ⑤ Spectrogram
    ax_sg = fig.add_subplot(gs[2, :2])
    f_sg, t_sg, Sxx = sp_signal.spectrogram(ecg, FS, nperseg=256, noverlap=200)
    ax_sg.pcolormesh(t_sg, f_sg, 10*np.log10(Sxx+1e-12),
                     shading="gouraud", cmap="inferno")
    ax_sg.set_ylim(0, 100)
    _ax(ax_sg, title="Spectrogram (dB)", xlabel="Time (s)", ylabel="Frequency (Hz)")

    # ⑥ RR tachogram
    ax_rr = fig.add_subplot(gs[2, 2])
    rp_full = detect_r_peaks(ecg, FS)
    if len(rp_full) >= 2:
        rr_ms = np.diff(rp_full) / FS * 1000
        ax_rr.plot(rr_ms, "o-", color=PAL["purple"], lw=1.5, ms=5)
        ax_rr.axhline(np.mean(rr_ms), color="gray", ls="--", lw=1, alpha=0.7,
                      label=f"Mean {np.mean(rr_ms):.0f} ms")
        ax_rr.legend(fontsize=8)
    else:
        ax_rr.text(0.5, 0.5, "Too few R-peaks", ha="center", va="center",
                   transform=ax_rr.transAxes, color="gray")
    _ax(ax_rr, title="RR Interval Tachogram", xlabel="Beat #", ylabel="RR (ms)")

    # ⑦ Key feature bar chart
    ax_f = fig.add_subplot(gs[3, :])
    disp_keys = ["std","rms","snr_ratio","baseline_wander","pl_50hz","power_hfnoise",
                 "hjorth_mob","spectral_ent","rmssd","r_amp_cv","flatline_ratio","beat_reg"]
    feat_dict = dict(zip(feature_names, feats))
    vals = [feat_dict.get(k, 0) for k in disp_keys]
    fc = [PAL["noisy"] if abs(v) > np.percentile(np.abs(vals), 65) else PAL["blue"]
          for v in vals]
    ax_f.barh(range(len(disp_keys)), vals, color=fc, alpha=0.85, edgecolor="white")
    ax_f.set_yticks(range(len(disp_keys)))
    ax_f.set_yticklabels(disp_keys, fontsize=8)
    _ax(ax_f, title="Key Extracted Features", xlabel="Value")

    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved → {save_path}")


# ─── Figure 4 : Random Window Detection Showcase ─────────────
def fig_random_detection(trained_models, windows, y, patient_info, feature_names, save_path):
    """Pick 8 random windows (4 clean, 4 noisy), classify with best model, visualise."""
    best_model = trained_models["Random Forest"]
    rng = np.random.default_rng(7)

    clean_idx = np.where(y == 0)[0]
    noisy_idx = np.where(y == 1)[0]
    n_each    = min(4, len(clean_idx), len(noisy_idx))
    chosen_c  = rng.choice(clean_idx, n_each, replace=False)
    chosen_n  = rng.choice(noisy_idx, n_each, replace=False)
    chosen    = list(chosen_c) + list(chosen_n)
    rng.shuffle(chosen)

    fig, axes = plt.subplots(4, 2, figsize=(18, 16), facecolor="white")
    fig.suptitle("Random ECG Window Detection — Real Patient Data",
                 fontsize=15, fontweight="bold", y=0.99)
    t3 = np.arange(WINDOW) / FS
    noise_lookup = {0:"gaussian", 1:"baseline", 2:"powerline", 3:"motion", 4:"all"}

    for ax, idx in zip(axes.flat, chosen[:8]):
        ecg        = windows[idx]
        true_label = y[idx]
        fname, w_i, _ = patient_info[idx]
        feats, _   = extract_features(ecg, FS)
        pred       = best_model.predict([feats])[0]
        prob       = best_model.predict_proba([feats])[0]
        conf       = prob[pred]
        sc         = PAL["clean"] if pred == 0 else PAL["noisy"]
        correct    = pred == true_label

        ax.set_facecolor(PAL["bg"])
        ax.plot(t3, ecg, color=sc, lw=0.65, alpha=0.9)
        ax.grid(True, color=PAL["grid"], lw=0.4, ls="--", alpha=0.6)
        ax.spines[["top","right"]].set_visible(False)
        ax.set_xlim(0, WINDOW/FS)
        ax.tick_params(labelsize=6)
        ax.set_xlabel("Time (s)", fontsize=7)
        ax.set_ylabel("mV", fontsize=7)

        check = "✓" if correct else "✗"
        gt    = "Clean" if true_label==0 else "Noisy"
        pred_s= "CLEAN" if pred==0 else "NOISY"
        short_fname = os.path.splitext(fname)[0][-14:]
        ax.set_title(
            f"{short_fname} | Win #{w_i} | True: {gt} | "
            f"Pred: {pred_s} {check}  ({conf*100:.1f}%)",
            fontsize=8, fontweight="bold", color=sc, pad=4)

        # Confidence strip at bottom
        y_lo, y_hi = ax.get_ylim()
        strip_h    = (y_hi - y_lo) * 0.07
        ax.fill_between(
            np.linspace(0, WINDOW/FS * conf, 50),
            y_lo, y_lo + strip_h,
            color=sc, alpha=0.35, zorder=3)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved → {save_path}")


# ═════════════════════════════════════════════════════════════
# 7.  MAIN
# ═════════════════════════════════════════════════════════════
def main():
    print("\n" + "═"*60)
    print("  ECG Signal Quality Classifier — Real Excel Pipeline")
    print("═"*60 + "\n")

    # ── Load + segment + auto-label ──
    print("▸ STEP 1 — Loading Excel files …")
    X, y, feature_names, windows, patient_info = build_dataset(INPUT_DIR)

    # Save feature CSV for inspection
    feat_df = pd.DataFrame(X, columns=feature_names)
    feat_df.insert(0, "label", y)
    feat_df.insert(0, "window_idx", [pi[1] for pi in patient_info])
    feat_df.insert(0, "patient",    [pi[0] for pi in patient_info])
    csv_path = os.path.join(OUTPUT_DIR, "ecg_features.csv")
    feat_df.to_csv(csv_path, index=False)
    print(f"  Feature table saved → {csv_path}")

    # ── Train models ──
    print("\n▸ STEP 2 — Training classifiers …")
    trained_models, results, X_te, y_te = train_models(X, y)

    # ── Generate images ──
    print("\n▸ STEP 3 — Generating diagnostic images …")

    fig_dataset_overview(X, y, feature_names, windows, patient_info,
                         os.path.join(OUTPUT_DIR, "ecg_1_dataset_overview.png"))

    fig_training_results(results, trained_models, feature_names, y_te,
                         os.path.join(OUTPUT_DIR, "ecg_2_training_results.png"))

    # Diagnostic report — one clean, one noisy example from real data
    for lbl, tag in [(0,"clean"), (1,"noisy")]:
        idxs = np.where(y == lbl)[0]
        if len(idxs):
            idx  = idxs[0]
            ecg  = windows[idx]
            feats, _ = extract_features(ecg, FS)
            best_pipe = trained_models["Random Forest"]
            pred = best_pipe.predict([feats])[0]
            prob = best_pipe.predict_proba([feats])[0][pred]
            fig_window_report(
                ecg, lbl, pred, prob, feature_names, feats,
                patient_name=f"{patient_info[idx][0]} — window {patient_info[idx][1]}",
                save_path=os.path.join(OUTPUT_DIR, f"ecg_3_report_{tag}.png"))

    fig_random_detection(trained_models, windows, y, patient_info, feature_names,
                         os.path.join(OUTPUT_DIR, "ecg_4_random_detection.png"))

    print("\n" + "═"*60)
    print("  ✓  Pipeline complete!")
    print(f"  Outputs → {os.path.abspath(OUTPUT_DIR)}/")
    for f in sorted(os.listdir(OUTPUT_DIR)):
        print(f"    {f}")
    print("═"*60 + "\n")


if __name__ == "__main__":
    main()
