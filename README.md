# ECG Signal Quality Detection using Machine Learning

## 📌 Overview

This project focuses on analyzing ECG (Electrocardiogram) signals and classifying signal quality using machine learning techniques. It performs preprocessing, feature extraction, model training, and detection of ECG signal quality (clean vs noisy). It includes a standalone detection script and a Flask-based web frontend for uploading ECG files and visualizing the results.

---

## 🚀 Features

* ECG signal preprocessing and cleaning
* Feature extraction from ECG time-series data
* Machine learning model training (Random Forest, Gradient Boosting, SVM)
* Signal quality classification (Clean / Noisy)
* Visualization of ECG signals and detection results
* **Web-based dashboard** for easy result visualization and file uploads

---

## 📂 Project Structure

```
ecgqualityassessment/
│
├── app.py                 # Flask web application frontend
├── ecg_pipeline.py        # Model training pipeline
├── ecg_detect.py          # Standalone detection script
├── requirements.txt       # Project dependencies
├── README.md              # Project documentation
│
├── static/                # Static files for the web frontend (HTML, CSS, JS)
├── web_uploads/           # Directory for user-uploaded ECG files via the web app
│
├── ecg_data/              # Training dataset directory
│   └── *.csv / *.xlsx     # Input ECG data for training
│
└── ecg_outputs/           # Generated outputs
    ├── best_ecg_model.pkl # Trained machine learning model
    ├── detection_*.csv    # Output predictions
    ├── *.png              # Visualizations and diagnostic reports
    └── ecg_features.csv   # Extracted feature sets
```

---

## ⚙️ Requirements

To install the project dependencies, ensure you have Python installed, then run:

```bash
pip install -r requirements.txt
```

---

## ▶️ How to Run

### Step 1: Train the Model

Ensure your `.xlsx` or `.csv` files are inside the `ecg_data` folder, then run the pipeline to extract features and train the model:

```bash
python ecg_pipeline.py
```

### Step 2: Use the Web Interface (Recommended)

Start the Flask application:

```bash
python app.py
```
Then, open your browser and navigate to `http://127.0.0.1:5000` to upload ECG files and see the quality assessment visually mapped out.

### Alternative: Run Standalone Detection

You can also run predictions without the web interface using the Python script. Ensure you update `NEW_PATIENT_FILE` inside the script to point to the file you want to test.

```bash
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

* Avoid hardcoded paths in new scripts. Use dynamic paths (e.g., `os.getcwd()`) like those configured in `app.py`. Ensure that any hardcoded paths in `ecg_detect.py` are properly changed to match your local file system before executing it directly.
* A compatible `scikit-learn` version is required to load the saved `best_ecg_model.pkl`.

---

## 📈 Future Enhancements

* Deep learning models (CNN/LSTM)
* Real-time ECG signal monitoring
* Multi-class ECG abnormality detection

---

## 👩‍💻 Author

Sivapriyadarsini

---

## 📜 License

This project is developed for academic purposes.
