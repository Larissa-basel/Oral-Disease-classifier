# 🦷 Oral Disease Classifier — CNN Model Comparison

A 6-class oral disease image classifier comparing a custom-built CNN against transfer learning with EfficientNetB0 and ResNet50 (each evaluated frozen and fine-tuned). Includes an interactive Streamlit app to run all three models on an uploaded photo side by side.

**🔗 Live demo:** [oral-disease-classifier-eduydiaum8emgapgbwspdk.streamlit.app](https://oral-disease-classifier-eduydiaum8emgapgbwspdk.streamlit.app/)

## Results

| Model | Test Accuracy | Macro F1 |
|---|---|---|
| Custom CNN | 67.8% | 0.625 |
| EfficientNetB0 (frozen) | 79.2% | 0.762 |
| EfficientNetB0 (fine-tuned) | 80.7% | 0.787 |
| ResNet50 (frozen) | 78.7% | 0.748 |
| ResNet50 (fine-tuned) | **82.8%** | **0.817** |

Transfer learning outperformed the custom CNN by roughly 15 points of accuracy, and fine-tuning the top layers added a further 2–4 points over frozen feature extraction on both pretrained backbones.

## How it was built

- **Custom CNN:** sequential conv blocks with batch normalization, global average pooling, and dense layers, trained from scratch.
- **Transfer learning:** EfficientNetB0 and ResNet50 with ImageNet weights, each trained in two phases — a frozen feature-extraction phase, then fine-tuning of the last 30 layers at a low learning rate.
- **Class imbalance:** handled with `compute_class_weight` during training.
- **Hyperparameter tuning:** a sequential coordinate search across learning rate, dropout, and dense units.

## Tech stack

Python · TensorFlow / Keras · Streamlit · scikit-learn · Google Colab

## Project structure

```
oral-disease-classifier/
├── app.py                      # Streamlit app
├── new-ver-oral-diseas.ipynb   # Model training notebook
├── requirements.txt
└── saved_models/
    ├── custom_cnn.keras
    ├── efficientnetb0.keras
    ├── resnet50.keras
    ├── metadata.json
    └── class_indices.json
```

## Run it locally

```bash
git clone https://github.com/YOUR_USERNAME/oral-disease-classifier.git
cd oral-disease-classifier
pip install -r requirements.txt
streamlit run app.py
```

## Dataset

_it was imported from Kaggle (shown in the first cell of the notebook)._
