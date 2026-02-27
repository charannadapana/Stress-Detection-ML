# 🧠 Stress Detection Using Machine Learning

A Machine Learning-based web application that detects whether a given text indicates stress or not using NLP techniques and classification models.

---

## 🚀 Project Overview

This project analyzes textual input and predicts whether the person is **Stressed** or **Not Stressed**.

It uses TF-IDF vectorization and multiple machine learning models, compares their performance, and selects the best model for prediction.

---

## ✨ Features

- 📊 Label Distribution Visualization
- 🔤 TF-IDF Text Vectorization
- 🤖 Multiple ML Models:
  - Logistic Regression
  - Linear SVM
  - Naive Bayes
- 📈 Model Accuracy Comparison Graph
- 📉 Confusion Matrix Visualization
- 🌐 Gradio Web Interface
- ⚙️ Clean Modular Project Structure

---

## 🛠 Tech Stack

- Python
- Scikit-learn
- Pandas
- Matplotlib
- Gradio

---

## 📂 Project Structure


StressDetectionProject/
│
├── data/
│ └── stress_data.csv
│
├── src/
│ ├── data_loader.py
│ ├── preprocessing.py
│ ├── train.py
│ ├── evaluate.py
│ └── app.py
│
├── main.py
├── requirements.txt
└── README.md


---

## 📊 Machine Learning Workflow

1. Data Loading
2. Train-Test Split (Stratified)
3. TF-IDF Vectorization
4. Model Training
5. Hyperparameter Tuning (GridSearchCV)
6. Model Evaluation (Accuracy & F1 Score)
7. Best Model Selection
8. Deployment via Gradio Interface

---

## ▶️ How To Run The Project

### 1️⃣ Clone the Repository


git clone https://github.com/charannadapana/Stress-Detection-ML.git

cd Stress-Detection-ML


### 2️⃣ Create Virtual Environment (Optional but Recommended)


python -m venv venv
venv\Scripts\activate


### 3️⃣ Install Dependencies


pip install -r requirements.txt


### 4️⃣ Run the Application


python main.py


The Gradio interface will open in your browser.

---

## 🧪 Example Test Sentences

"i am happy now"
"I feel completely overwhelmed with my work and I don’t know how to manage everything."
---

## 🎯 Future Improvements

- Use Deep Learning (LSTM / BERT)
- Deploy on Hugging Face / Render
- Improve dataset size
- Add real-time sentiment tracking
