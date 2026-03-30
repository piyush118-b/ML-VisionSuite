# ML Vision Suite

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://ml-visionsuitegit-ztf92b7mnjyg3tveat5uqd.streamlit.app/)

Welcome to the **ML Vision Suite**—a unified dashboard designed to explore the intersection of sight, sound, and data through modern machine learning. This suite brings together complex inference models for real-time video analysis, audio processing, and predictive analytics into a single, high-performance interface.

**[View the Live Demo](https://ml-visionsuitegit-ztf92b7mnjyg3tveat5uqd.streamlit.app/)**

---

## Overview
The ML Vision Suite is built to handle multiple types of data input, providing immediate feedback across five distinct modules:
- **Text Analysis**: Sentiment and tone detection using state-of-the-art Transformers.
- **Image Classification**: Species identification for biological samples.
- **Audio Analysis**: Character transcription from audio CAPTCHAs.
- **Video Processing**: Frame-by-frame monitoring for fire and hazard detection.
- **Market Analytics**: Discovering patterns in retail data through association rules.

---

## Recent Improvements
We recently performed a complete technical sweep to move the project from a prototype to a polished application. These updates include:
- **UI & Contrast**: Re-engineered the CSS system to ensure high legibility. Text areas, progress bars, and labels now maintain clear visibility even on high-brightness displays or within varying browser themes.
- **Performance**: Optimized video and image processing by implementing efficient frame sampling and pre-processing routines.
- **Stability**: Added comprehensive error handling to manage model loading and inference failures gracefully without crashing the UI.
- **Environment**: Standardized dependencies to ensure a consistent experience across Windows and macOS environments.

---

## Project Structure
```text
ML-VisionSuite/
├── app.py                      # Main application logic and UI routing
├── pkls/                       # Pre-trained model weights and architectures
│   ├── best_video_fire_model.pkl
│   ├── optimized_champion_package.pkl
│   ├── audio_captcha_model.pkl
│   └── Market_Basket_Model3.pkl
├── requirements.txt            # Project dependencies and versions
└── README.md                   # Project documentation
```

---

## Getting Started

### Installation
To run the suite on your local machine, follow these steps:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/piyush118-b/ML-VisionSuite.git
   cd ML-VisionSuite
   ```

2. **Configure your environment**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # macOS/Linux
   # or .venv\Scripts\activate on Windows
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

### Running Locally
Launch the application using the Streamlit CLI:
```bash
streamlit run app.py
```
By default, the application will be accessible at **[http://localhost:8501](http://localhost:8501)**.

---

---

## Detailed Model Insights

The ML Vision Suite leverages a variety of specialized architectures to provide high-accuracy inference across different data modalities.

### 🔥 Fire Detection (Video Analytics)
*   **Architecture**: MobileNetV2 (Lightweight CNN)
*   **Framework**: TensorFlow / Keras
*   **Mechanism**: The model performs frame-by-frame binary classification. To optimize for real-time performance in web environments, it employs a **temporal sampling strategy** (sampling every 5th frame by default).
*   **Core Logic**: Each sampled frame is resized to 224x224 and normalized. The MobileNetV2 backbone extracts features that are then passed through a global average pooling layer and a dense sigmoid output. The suite identifies the "Peak Confidence" frame and provides a cumulative safety verdict.

### 🍄 Mushroom Classifier (Computer Vision)
*   **Architecture**: Optimized Deep Convolutional Neural Network (CNN)
*   **Framework**: TensorFlow / Keras
*   **Scope**: Trained to identify 9 distinct classes of mushrooms (e.g., *Amanita*, *Agaricus*, *Boletus*).
*   **Mechanism**: Processes images through multiple convolutional and max-pooling layers for spatial feature extraction. The final layer uses **Softmax activation** to provide a probability distribution across known species.
*   **Output**: Delivers the top species prediction along with a confidence histogram for the top 5 likely candidates.

### 🔊 Audio CAPTCHA Solver (Speech-to-Text)
*   **Architecture**: Deep CRNN (CNN + Bidirectional GRU)
*   **Framework**: PyTorch / Torchaudio
*   **Logic Pipeline**:
    - **Feature Extraction**: Converts raw audio waveforms into **Mel-Spectrograms**.
    - **Convolutional Layers**: Extracts acoustic patterns from the frequency-domain data.
    - **Recurrent Layers**: A Bidirectional GRU models the temporal dependencies of spoken characters.
    - **CTC Decoding**: Employs **Connectionist Temporal Classification** to map variable-length sequences to text without needing per-character alignment.
*   **Use Case**: Specifically designed for transcribing noisy alphanumeric audio CAPTCHAs.

### 🛒 Market Basket (Association Rule Mining)
*   **Algorithm**: FP-Growth (Frequent Pattern Growth)
*   **Framework**: MLxtend
*   **Technical Concept**: Unlike gradient-based learning, this uses a tree-based data structure to store transactional patterns efficiently.
*   **Key Metrics**:
    - **Support**: Frequency of item occurrence in the dataset.
    - **Confidence**: Likelihood of purchasing "Item B" when "Item A" is in the cart.
    - **Lift**: Measures the strength of the association (Lift > 1 implies a relationship stronger than random chance).
*   **Output**: Dynamic product recommendations based on frequent co-occurrence patterns.

### 📝 Sentiment Analysis (NLP)
*   **Architecture**: DistilBERT / RoBERTa (Transformer-based)
*   **Framework**: Hugging Face Transformers
*   **Mechanism**: Utilizes **Self-Attention mechanisms** to understand the bidirectional context of a sentence. It identifies the semantic weight of words to determine the emotional tone.
*   **Precision**: Fine-tuned on the SST-2 dataset, providing high sensitivity to complex linguistic structures and emotional modifiers.
*   **Output**: Sentiment labels (Positive/Negative/Neutral) with detailed probability scores.

---

---

## Troubleshooting

| Issue | Potential Cause | Recommended Fix |
| :--- | :--- | :--- |
| **Dependency Errors** | Missing packages | Ensure your virtual environment is active and run `pip install -r requirements.txt`. |
| **Port Conflicts** | Address already in use | Run `streamlit run app.py --server.port 8502` to use an alternative port. |
| **Pickle Errors** | Version mismatch | Use the provided `.venv` setup to ensure package versions match the model training environment. |
| **Display Issues** | Browser Cache | If you see invisible text, clear your browser cache or force a refresh (Cmd+Shift+R). |

---

## A Note on Security and Performance
- **Security**: This project uses Pickle files (`.pkl`) to store models. Only use models from this repository; loading untrusted pickles can execute harmful code.
-  **Performance**: All models are cached upon initial load. The first inference may take longer as the model is moved into memory, but subsequent runs will be significantly faster.

---

## Deployment
This project is optimized for deployment on **Streamlit Community Cloud**. To deploy your own version, simply push your repository to GitHub and connect it at [share.streamlit.io](https://share.streamlit.io/).

Created with care as a demonstration of modern multi-modal AI integration. 🚀
