import os
import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename

# Import core logic from ecg_pipeline
import ecg_pipeline

app = Flask(__name__, static_folder='static')

# Configuration
UPLOAD_FOLDER = os.path.join(os.getcwd(), 'web_uploads')
OUTPUT_FOLDER = os.path.join(os.getcwd(), 'ecg_outputs')
MODEL_PATH = os.path.join(OUTPUT_FOLDER, 'best_ecg_model.pkl')
ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xls'}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/static/<path:path>')
def send_static(path):
    return send_from_directory(app.static_folder, path)

@app.route('/outputs/<path:path>')
def send_outputs(path):
    return send_from_directory(OUTPUT_FOLDER, path)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

@app.route('/api/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            return run_detection(filepath, filename)
        except Exception as e:
            import traceback
            print(traceback.format_exc())
            return jsonify({'error': str(e)}), 500
            
    return jsonify({'error': 'File type not allowed'}), 400

def save_summary_image(signal, preds, fname, save_path):
    PAL = {"clean":"#27AE60", "noisy":"#E74C3C", "accent":"#2C3E50",
           "bg":"#F8F9FA",    "grid":"#DEE2E6",  "blue":"#3498DB"}
    FS = ecg_pipeline.FS
    WINDOW = ecg_pipeline.WINDOW
    
    fig, ax = plt.subplots(figsize=(15, 3), facecolor="white")
    t = np.arange(len(signal)) / FS
    ax.plot(t, signal, color="#BDC3C7", lw=0.5, alpha=0.6, zorder=1)

    for i, pred in enumerate(preds):
        start = i * WINDOW
        end   = start + WINDOW
        color = PAL["clean"] if pred == 0 else PAL["noisy"]
        ax.axvspan(start/FS, end/FS, alpha=0.25, color=color, zorder=2)

    ax.set_facecolor(PAL["bg"])
    ax.grid(True, color=PAL["grid"], lw=0.4, ls="--", alpha=0.6)
    ax.spines[["top","right"]].set_visible(False)
    ax.set_xlabel("Time (s)", fontsize=9)
    ax.set_ylabel("mV", fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight", facecolor="white")
    plt.close(fig)

def run_detection(filepath, original_filename):
    if not os.path.exists(MODEL_PATH):
        return jsonify({'error': 'Model not found. Please run ecg_pipeline.py first.'}), 404
        
    with open(MODEL_PATH, "rb") as f:
        saved_model = pickle.load(f)
    model = saved_model["model"]
    
    if filepath.endswith('.csv'):
        df = pd.read_csv(filepath, header=0, skiprows=[1])
    else:
        df = pd.read_excel(filepath, header=0, skiprows=[1])
        
    df.columns = [str(c).strip().strip("'").strip().lower() for c in df.columns]
    lead_col = ecg_pipeline.LEAD_COL.lower()
    if lead_col not in df.columns:
        candidates = [c for c in df.columns if c == "ii" or "lead ii" in c or c == "2"]
        if candidates: lead_col = candidates[0]
        else: return jsonify({'error': 'Lead II column not found'}), 400
            
    sig = df[lead_col].to_numpy(dtype=np.float64)
    sig = sig[~np.isnan(sig)]
    
    windows = ecg_pipeline.segment_windows(sig, ecg_pipeline.WINDOW)
    
    results = []
    preds = []
    for i, win in enumerate(windows):
        feats, _ = ecg_pipeline.extract_features(win, ecg_pipeline.FS)
        pred = model.predict([feats])[0]
        preds.append(pred)
        proba = model.predict_proba([feats])
        conf = (proba[:, 1] if proba.shape[1] > 1 else proba[:, 0])[0]
        
        t_start = i * ecg_pipeline.WINDOW / ecg_pipeline.FS
        t_end = t_start + ecg_pipeline.WINDOW / ecg_pipeline.FS
        
        results.append({
            'window': i + 1,
            'start_s': round(t_start, 1),
            'end_s': round(t_end, 1),
            'label': 'CLEAN' if pred == 0 else 'NOISY',
            'confidence': round(float(conf) * 100, 1)
        })
        
    # Generate Map Image
    map_filename = f"web_map_{original_filename.split('.')[0]}.png"
    map_path = os.path.join(OUTPUT_FOLDER, map_filename)
    save_summary_image(sig, preds, original_filename, map_path)
    
    return jsonify({
        'filename': original_filename,
        'total_windows': len(results),
        'results': results,
        'map_image': map_filename
    })


if __name__ == '__main__':
    app.run(debug=True, port=5000)
