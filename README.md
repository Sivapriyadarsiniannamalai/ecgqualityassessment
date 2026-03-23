# ECG Signal Quality Detection using Machine Learning

## 📌 Overview

This project focuses on analyzing ECG (Electrocardiogram) signals and classifying signal quality using machine learning techniques. It performs preprocessing, feature extraction, model training, and detection of ECG signal quality (clean vs noisy).

---

## 🚀 Features

* ECG signal preprocessing and cleaning
* Feature extraction from ECG time-series data
* Machine learning model training (Random Forest)
* Signal quality classification (Clean / Noisy)
* Visualization of ECG signals and detection results

---

## 📂 Project Structure

```
ecg_project_op/
│
├── ecg_pipeline.py        # Model training pipeline
├── ecg_detect.py          # Detection script
├── BBB_228.csv            # Sample ECG input file
│
├── ecg_data/              # Training dataset
│   ├── *.csv
│
├── ecg_outputs/           # Generated outputs
│   ├── best_ecg_model.pkl
│   ├── detection_*.csv
│   ├── *.png
│   └── ecg_features.csv
```

---

## ⚙️ Requirements

Install dependencies:

```
pip install numpy scipy scikit-learn matplotlib pandas openpyxl tqdm
```

---

## ▶️ How to Run

### Step 1: Train the model

```
python ecg_pipeline.py
```

### Step 2: Run detection

```
python ecg_detect.py
```

---

## 📊 Output

The project generates:

* Trained model (`best_ecg_model.pkl`)
* Detection results (`detection_*.csv`)
* Extracted features (`ecg_features.csv`)
* Visualization graphs (`.png`)

---

## ⚠️ Important Notes

* Ensure input file (`BBB_228.csv`) is in the project directory
* Avoid hardcoded paths like `G:\`, use dynamic paths instead
* Compatible scikit-learn version may be required for loading saved model

---

## 📈 Future Enhancements

* Deep learning models (CNN/LSTM)
* Real-time ECG signal monitoring
* Web-based dashboard for visualization
* Multi-class ECG abnormality detection

---

## 👩‍💻 Author

Sivapriyadarsini

---

## 📜 License

This project is developed for academic purposes.
