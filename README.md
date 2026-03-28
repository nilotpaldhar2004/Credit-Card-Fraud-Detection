# 🛡️ FinSec Shield: Real-Time Credit Card Fraud Detection

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)
![Pandas](https://img.shields.io/badge/Pandas-2.0+-150458.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.2+-F7931E.svg)
![Plotly](https://img.shields.io/badge/Plotly-5.0+-3F4F75.svg)
![LightGBM](https://img.shields.io/badge/LightGBM-4.0+-orange.svg)
![Vercel](https://img.shields.io/badge/Frontend-Vercel-black.svg)
![Render](https://img.shields.io/badge/Backend-Render-purple.svg)

An end-to-end, production-ready Machine Learning microservice that detects fraudulent credit card transactions in real-time. This project encompasses the entire ML lifecycle: from exploratory data analysis and handling severe class imbalance, to model training, footprint optimization, and deploying a distributed cloud architecture.

### **🔗 [Live Interactive Dashboard](https://finsec-dashboard-silk.vercel.app/)** *(Note: The AI backend is hosted on Render's free tier. If the server has been inactive, the very first inference may take 30-50 seconds to wake up. Subsequent inferences execute in <50ms).*

---

## 🏗️ System Architecture (Microservice)

To replicate enterprise-grade deployments, this system is split into two distinct environments:

1. **The Brain (Backend API on Render):** A robust FastAPI server hosting a highly optimized **31 KB LightGBM model**. It handles the heavy tensor math, feature validation via Pydantic, and returns a strict JSON decision payload.
2. **The Face (Frontend UI on Vercel):** A zero-dependency, vanilla HTML/CSS/JS enterprise dashboard. Hosted on Vercel's edge network for lightning-fast global delivery, it securely routes transaction payloads to the Render API.

---

## 🧠 Machine Learning Engineering

* **Dataset:** Highly imbalanced credit card transaction dataset (fraud accounts for <0.2% of data).
* **Feature Engineering:** Processed 30 features (Time, Amount, and PCA-transformed variables V1-V28). 
* **Leakage Prevention:** Strictly isolated training and validation sets to prevent data leakage during scaling and sampling techniques.
* **Model Optimization:** Trained a Gradient Boosted Tree (LightGBM) optimized specifically for inference speed and high Recall to minimize false negatives in fraud detection. 

---

## 💻 Tech Stack

* **Machine Learning:** LightGBM, Pandas, Scikit-Learn, Joblib
* **Backend:** Python, FastAPI, Uvicorn, Pydantic
* **Frontend:** HTML5, CSS3, Vanilla JavaScript (Fetch API)
* **Deployment:** Vercel (Frontend), Render (Backend API)

---

## 🚀 Local Installation & Setup

Want to run this system on your local machine?

### 1. Clone the repository

```bash
git clone https://github.com/nilotpaldhar2004/Credit-Card-Fraud-Detection.git
cd Credit-Card-Fraud-Detection
```

### 2. Create virtual environment & install dependencies

```bash
python -m venv venv
```

**Activate environment:**

Mac/Linux:

```bash
source venv/bin/activate
```

Windows:

```bash
venv\Scripts\activate
```

**Install dependencies:**

```bash
pip install -r requirements.txt
```

### 3. Run backend server

```bash
python app.py
```

### 4. Open dashboard

Go to:

```
http://127.0.0.1:8500
```

---

## 📡 API Reference

### POST /predict

Evaluates a 30-feature transaction for fraud detection.

### Request (JSON)

```json
{
  "Time": 1205.0,
  "V1": -0.8, "V2": 0.5, "V3": 1.2, "V4": 0.1, "V5": -0.3,
  "V6": 0.0, "V7": 0.8, "V8": -0.1, "V9": 0.4, "V10": -0.2,
  "V11": 0.5, "V12": 0.1, "V13": -0.5, "V14": -1.2, "V15": 0.3,
  "V16": 0.4, "V17": 0.8, "V18": -0.2, "V19": 0.5, "V20": 0.1,
  "V21": 0.0, "V22": 0.1, "V23": -0.2, "V24": 0.3, "V25": 0.1,
  "V26": -0.1, "V27": 0.0, "V28": 0.0,
  "Amount": 250.00
}
```

### Response (JSON)

```json
{
  "status": "success",
  "fraud_probability": 0.0034,
  "is_fraud": false,
  "action": "APPROVE",
  "latency_ms": 42.1
}
```

---

## 👨‍💻 Author

**Nilotpal Dhar**  
Computer Science Student (AOT)  
Specializing in Machine Learning, Model Deployment, and API Development
