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

## Detailed Module Documentation

### Fire Detection (Video)
This module uses a MobileNetV2 architecture specifically tuned to detect fire and smoke. To maintain performance, the application samples every 5th frame, providing a real-time confidence score and identifying the exact timestamp of peak activity.

### Mushroom Classifier (Image)
A Vision-based CNN processes uploaded photos to identify mushroom species. It provides a ranked list of the top 5 most likely candidates, helping users differentiate between species with high confidence scores.

### Audio CAPTCHA Solver
Our audio solving module combines Convolutional and Recurrent neural networks (CRNN) with CTC decoding. It processes raw audio (WAV/MP3), extracts Mel-spectrograms, and transcribes spoken characters into a clear text output.

### Market Basket Analysis
Based on the FP-Growth algorithm, this tool analyzes retail datasets to find strong associations between products. It calculates Confidence and Lift scores to help understand which items are most likely to be purchased together.

### Sentiment Analysis
Leveraging the Hugging Face Transformers library (DistilBERT), this module interprets the emotional weight and tone of text inputs. It provides a score reflecting the model's confidence in identifying positive, negative, or neutral sentiments.

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
