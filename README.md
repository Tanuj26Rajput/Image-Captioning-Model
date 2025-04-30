# 🖼️ Image Captioning with Deep Learning

This project implements an **Image Captioning** system using deep learning techniques. It uses a combination of CNN (InceptionV3) for feature extraction and LSTM for generating textual captions, enhanced with **beam search decoding** for better output quality. The final model is wrapped in a **Gradio web interface** for easy user interaction.

## 📂 Dataset

- **Flickr8k** Dataset
- Contains 8,000 images with 5 captions each
- Captions file format: `image_name#<caption_id>, caption`

## 📌 Features

- Image preprocessing and caption tokenization
- Feature extraction using **InceptionV3**
- Sequence modeling using **LSTM**
- Caption generation using **Beam Search**
- Simple **Gradio Web App** for interaction

---

## 🛠️ Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Tanuj26Rajput/image-captioning-app.git
   cd image-captioning-app

## 🚀 How It Works

1. **Caption Preprocessing:**
   - Add `<start>` and `<end>` tokens
   - Tokenize and convert to sequences

2. **Feature Extraction:**
   - Use pretrained **InceptionV3** CNN
   - Extract 2048-D features from the penultimate layer

3. **Model Architecture:**
   - CNN (image features) + Embedding + LSTM (text features)
   - Dense layer to predict next word

4. **Training:**
   - Trained using a custom data generator
   - Optimized with `categorical_crossentropy`

5. **Prediction:**
   - Generates captions using **beam search** (default beam width = 3)

