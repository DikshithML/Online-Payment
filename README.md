# 🏦 Online Payment Fraud Detection

![Python](https://img.shields.io/badge/Python-3.9-blue.svg)
![Flask](https://img.shields.io/badge/Flask-2.3-lightgrey)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

A machine learning-based web application to detect fraudulent online payments. Built with Flask, this project uses an XGBoost classifier trained on real-world financial transaction data to predict whether a transaction is **legitimate** or **fraudulent**.

---

## 🚀 Features

- 🧠 Predict fraud based on transaction inputs
- 📊 Visualizations using Matplotlib & Seaborn
- 📁 CSV-based data processing (chunked)
- ⚡ Real-time response with XGBoost
- 🌐 Simple and clean web UI

---

## 🛠 Tech Stack

| Layer        | Technology |
|--------------|------------|
| Backend      | Python, Flask |
| ML/Modeling  | XGBoost, Scikit-learn |
| Data Handling| Pandas, NumPy |
| Visualization| Matplotlib, Seaborn |
| Frontend     | HTML5, CSS3 (Jinja2 templates) |

---

## 📂 Project Structure

```
Online-Payment/
├── app.py                 # Flask main application
├── fraud_detection.py     # ML logic & prediction code
├── PS_log.csv             # Sample dataset
├── templates/             # HTML templates
├── static/                # CSS, JS, images
├── requirements.txt       # Python dependencies
├── README.md              # This file
└── assets/                # (Optional folder for extras)
```

---

## 🧠 Machine Learning Model

The model is trained on the **PaySim Fraud Detection dataset**:  
[Kaggle Dataset](https://www.kaggle.com/datasets/ntnu-testimon/paysim1)

- Binary Classification using `XGBClassifier`
- Trained on simulated mobile money transactions
- Features include transaction type, amount, origin/destination balances

---

## 💻 How to Run Locally

```bash
# Step 1: Clone the repository
git clone https://github.com/DikshithML/Online-Payment.git
cd Online-Payment

# Step 2: Create and activate virtual environment
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Step 3: Install dependencies
pip install -r requirements.txt

# Step 4: Run the app
python app.py
```

🌐 Visit: `http://localhost:5000`

---

## ⚠️ Notes

- Make sure `PS_log.csv` is present in the root directory
- If using full dataset, rename or adjust filename in `app.py`
- You can extend this to use live APIs or integrate a frontend framework

---

## 📜 License

This project is licensed under the **MIT License**.  
Feel free to use, modify, and share!

---

> By Dikshith ML
