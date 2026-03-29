# 🧠 ML Vision Suite

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://ml-visionsuitegit-ztf92b7mnjyg3tveat5uqd.streamlit.app/)

Welcome to the **ML Vision Suite**! This is a project I built to bring together different "senses" of AI—sight, sound, and data—into one beautiful, easy-to-use dashboard. 

Whether it's spotting fire in a video or solving an audio puzzle, this suite shows what's possible when modern AI meets elegant design.

---

## ✨ What’s Inside?

### 🔥 Keeping an Eye on Fire
I built this to help detect fire in real-time. Whether you're uploading an image or a video, the AI scans every frame to alert you of potential danger.
*   **Video**: Real-time monitoring for smoke and flames.
*   **Image**: Quick checks for high-risk visuals.

### 🍄 The Forest Guide (Mushroom Classifier)
Ever wondered if that mushroom you found is safe? While you should always be careful, this tool uses a "vision transformer" (a very smart AI architecture) to help identify species and tell you if they're edible or poisonous.

### 🔊 Solving Audio Puzzles
The **Audio CAPTCHA Solver** is one of the more unique parts of this suite. It "listens" to audio security challenges and transcribes them instantly using a mix of Convolutional and Recurrent neural networks.

### 🛒 Understanding Shop Talk (Market Basket)
Retail is all about patterns. This tool looks at shopping carts and figures out which products people like to buy together, using the **Apriori algorithm**. It's great for understanding customer behavior.

### 📝 Reading the Room (Sentiment Analysis)
Finally, we have a tool that understands feelings. Paste any text, and the AI will tell you if the vibe is positive, negative, or just neutral. It’s perfect for gauging feedback or social media tone.

---

## 🛠️ The "Brain" Behind the Beauty

The suite looks premium, but under the hood, it’s powered by some heavy hitters:
*   **Visuals**: Built with **TensorFlow** and **PyTorch**.
*   **Listening**: Powered by **Librosa** and **Torchaudio**.
*   **Smart Analytics**: Driven by **MLxtend** and **Pandas**.
*   **The Look**: Hand-crafted with **Streamlit** and custom CSS for that "PresenceX" premium feel.

---

## 🚀 Get it Running on Your Machine

1.  **Grab the code**:
    ```bash
    git clone https://github.com/piyush118-b/ML-VisionSuite.git
    cd ML-VisionSuite
    ```

2.  **Set up your space**:
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # Or .venv\Scripts\activate on Windows
    ```

3.  **Install the "Brains"**:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Launch the Suite**:
    ```bash
    streamlit run app.py
    ```

---

## 🌍 Take it to the Web

This app is live and ready to explore! 
**[View the Live Demo here](https://ml-visionsuitegit-ztf92b7mnjyg3tveat5uqd.streamlit.app/)**

It was built specifically to shine on **Streamlit Community Cloud**. 
Just push your project to GitHub (which we've already set up!), connect it at [share.streamlit.io](https://share.streamlit.io/), and you're live!

---

Hope you enjoy exploring the ML Vision Suite as much as I enjoyed building it! 🚀
