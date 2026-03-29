# 🧠 ML Vision Suite

A unified, multi-modal machine learning dashboard built with **Streamlit** and **PresenceX-inspired bright design**. This suite combines state-of-the-art computer vision, audio processing, and data analytics into a single, cohesive experience.

---

## ✨ Features

### 🔥 Fire Detection
- **Video Analysis**: Real-time detection of fire in video streams using optimized deep learning models.
- **Image Analysis**: Upload images for instant classification and risk assessment.

### 🍄 Mushroom Classification
- High-accuracy identification of mushroom species.
- Provides edible vs. poisonous guidance using a custom-trained vision transformer.

### 🔊 Audio CAPTCHA Solver
- Advanced CRNN (Convolutional Recurrent Neural Network) architecture.
- Real-time transcription of audio-based security challenges.

### 🛒 Market Basket Analysis
- Association rule mining for retail intelligence.
- Helps identify frequent itemsets and product relationships using the Apriori algorithm.

### 📝 Sentiment Analysis
- Natural Language Processing to detect emotional tone in text.
- Categorizes sentiment into positive, negative, or neutral with confidence scoring.

---

## 🛠️ Technology Stack

- **Frontend**: [Streamlit](https://streamlit.io/) (Custom CSS Injection)
- **Deep Learning**: [TensorFlow](https://www.tensorflow.org/), [PyTorch](https://pytorch.org/)
- **Computer Vision**: [OpenCV](https://opencv.org/)
- **Audio Processing**: [Librosa](https://librosa.org/), [Torchaudio](https://pytorch.org/audio/stable/index.html)
- **Data Science**: [Numpy](https://numpy.org/), [Pandas](https://pandas.pydata.org/), [MLxtend](http://rasbt.github.io/mlxtend/)
- **Design System**: PresenceX (Bright, airy, and premium typography)

---

## 🚀 Installation & Local Run

1. **Clone the repository**:
   ```bash
   git clone https://github.com/piyush118-b/ML-VisionSuite.git
   cd ML-VisionSuite
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Application**:
   ```bash
   streamlit run app.py
   ```

---

## 🌍 Deployment

This app is optimized for **Streamlit Community Cloud**:
1. Push your changes to GitHub.
2. Connect your repo at [share.streamlit.io](https://share.streamlit.io/).
3. Deploy directly from the `main` branch.

---

## 📄 License
MIT License. See `LICENSE` for details.
