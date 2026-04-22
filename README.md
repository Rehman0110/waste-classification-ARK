# 🗑️ Smart Waste Segregation System using CNN

An end-to-end deep learning project that classifies waste images into **Recyclable** and **Organic** categories using MobileNetV2 Transfer Learning and Keras 3.

---

## 📂 Project Structure

```
smart_waste_segregation/
├── data/
│   ├── raw/                  # Original downloaded dataset
│   └── processed/            # Augmented/split dataset
├── models/
│   ├── cnn_model.h5          # Custom CNN (saved after training)
│   └── mobilenet_model.h5    # MobileNetV2 model (saved after training)
├── app/
│   └── streamlit_app.py      # Streamlit web application
├── utils/
│   ├── data_loader.py        # Dataset loading & augmentation
│   ├── model_builder.py      # CNN & Transfer Learning model definitions
│   ├── trainer.py            # Training pipeline
│   ├── evaluator.py          # Evaluation metrics & confusion matrix
│   └── voice_output.py       # Text-to-speech output
├── static/
│   ├── css/                  # Stylesheets
│   ├── js/                   # JavaScript
│   └── uploads/              # Uploaded images (temp)
├── templates/                # HTML templates (for Flask fallback)
├── main.py                   # Entry point: train + evaluate
├── webcam_demo.py            # Real-time webcam classification (OpenCV)
├── requirements.txt          # All dependencies
└── README.md
```

---

## 🚀 How to Run

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Download Dataset
Use the **TrashNet** dataset or any waste image dataset organized as:
```
data/
├── recyclable/    (paper, cardboard, plastic, metal, glass)
├── organic/       (food scraps, leaves, wood)
└── hazardous/     (batteries, chemicals, electronics)
```

**Option A – TrashNet** (auto-download script):
```bash
python utils/download_dataset.py
```

**Option B – Use the built-in synthetic demo data** (for testing without a dataset):
```bash
python utils/create_demo_data.py
```

### 3. Train the Model
```bash
python main.py --model mobilenet --epochs 20
```
Options:
- `--model cnn` → Custom CNN from scratch
- `--model mobilenet` → MobileNetV2 transfer learning (recommended)
- `--epochs N` → Number of training epochs

### 4. Run the Streamlit Web App
```bash
streamlit run app/streamlit_app.py
```
Then open **http://localhost:8501** in your browser.

### 5. Run Webcam Real-Time Demo (Bonus)
```bash
python webcam_demo.py
```
Press `Q` to quit.

---

## 🧠 Model Architecture

### Custom CNN
```
Input (224×224×3)
→ Conv2D(32) + ReLU + MaxPool
→ Conv2D(64) + ReLU + MaxPool
→ Conv2D(128) + ReLU + MaxPool
→ Flatten
→ Dense(256) + ReLU + Dropout(0.5)
→ Dense(3) + Softmax
```

### MobileNetV2 (Transfer Learning)
```
MobileNetV2 (ImageNet weights, frozen base)
→ GlobalAveragePooling2D
→ Dense(128) + ReLU + Dropout(0.3)
→ Dense(3) + Softmax
```


## 📊 Output

- **Accuracy/Loss plots** saved to `models/mobilenet_v2_training_plots.png`
- **Confusion matrix** saved to `models/mobilenet_v2_cm.png`

---

## 🔊 Voice Output

When a prediction is made, the system speaks the result aloud:
> *"This is recyclable waste. Please send it to the recycling bin."*

---

## 📦 Classes & Suggestions

| Class | Icon | Suggestion |
|-------|------|-----------|
| Recyclable | ♻️ | Send to recycling bin |
| Organic | 🌿 | Compost this waste |


## 🛠️ Tech Stack

- **Deep Learning**: TensorFlow / Keras
- **Transfer Learning**: MobileNetV2
- **Web App**: Streamlit
- **Computer Vision**: OpenCV
- **Voice Output**: pyttsx3
- **Visualization**: Matplotlib, Seaborn, Plotly
- **Data Handling**: NumPy, Pandas, Pillow
