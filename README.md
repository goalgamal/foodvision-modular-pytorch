# FoodVision: Modular PyTorch Image Classification

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)
![Gradio](https://img.shields.io/badge/Gradio-Demo-pink)

A modular computer vision project demonstrating two core workflows: training a custom CNN (TinyVGG) from scratch and deploying a State-of-the-Art model (EfficientNetB2) using Transfer Learning.

---

## 🚀 Live Demo
The model is deployed and serving predictions via Hugging Face Spaces.
<a href="https://huggingface.co/spaces/goalgamal/foodvision-modular-pytorch">
  <img src="https://huggingface.co/datasets/huggingface/badges/raw/main/open-in-hf-spaces-lg.svg" alt="Open in Spaces">
</a>

---

## 📂 Project Structure

This project follows a modular "script mode" structure, separating data processing, model architecture, and training logic into distinct files.

```text
├── data/                       # Dataset storage (ignored by git)
├── models/                     # Saved models (ignored by git)
├── app.py                      # Gradio web application for deployment
├── data_setup.py               # Data loading and preprocessing
├── engine.py                   # Training and testing loops
├── model.py                    # EfficientNetB2 feature extractor builder
├── model_builder.py            # TinyVGG custom architecture
├── predictions.py              # Inference utilities for single images
├── train.py                    # Main training script
├── utils.py                    # Utility functions (saving models, etc.)
└── requirements.txt            # Project dependencies

```

## 🛠 Features

* **Modular Design**: Code is decoupled into `engine`, `data_setup`, and `model_builder` for better maintainability.
* **Custom Architecture**: Implements the **TinyVGG** architecture from scratch for basic classification tasks.
* **Transfer Learning**: Utilizes a pre-trained **EfficientNetB2** for high-accuracy food classification (101 classes).
* **Interactive Demo**: Includes a `Gradio` web interface (`app.py`) to test the model in real-time.

## 🚀 Installation

1. **Clone the repository**
```bash
git clone [https://github.com/YOUR_USERNAME/foodvision-modular.git](https://github.com/YOUR_USERNAME/foodvision-modular.git)
cd foodvision-modular

```


2. **Install dependencies**
```bash
pip install -r requirements.txt

```



## 💻 Usage

### 1. Training a Model (TinyVGG)

To train the custom TinyVGG model on your dataset (default: Pizza, Steak, Sushi), run the training script. This script handles data loading, model creation, and the training loop.

```bash
python train.py

```

*Note: Ensure your training data is located in `data/pizza_steak_sushi/` or update the directory paths in `train.py`.*

### 2. Running the Deployment App

To launch the FoodVision Big web interface (using EfficientNetB2):

```bash
python app.py

```

This will launch a local Gradio server where you can upload images to be classified into 101 food categories.

*Note: The app requires the pre-trained weights file `09_pretrained_effnetb2_feature_extractor_food101_20_percent.pth` to be present in the root directory.*

## 🧠 Model Details

### TinyVGG (Custom)

* **Input**: 64x64 RGB Images
* **Architecture**: 2 Convolutional Blocks followed by a Classifier head.
* **Purpose**: Lightweight training on smaller datasets.

### EfficientNetB2 (Transfer Learning)

* **Input**: Resized and Normalized via `weights.transforms()`.
* **Architecture**: Pre-trained EfficientNetB2 with a frozen base and a custom classifier head for 101 classes.

## 📚 Acknowledgements

This code is based on the [Learn PyTorch for Deep Learning](https://www.learnpytorch.io/) course by Daniel Bourke.

```
