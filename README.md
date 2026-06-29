# Water Quality Classification using Supervised Learning Algorithms

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-orange)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Supervised-green)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)
![License](https://img.shields.io/badge/License-MIT-yellow)

## 📌 Project Overview

This project focuses on **Water Quality Classification** using supervised machine learning algorithms. The main goal is to classify water samples as **Safe** or **Not Safe** based on different chemical and biological water quality parameters.

The project applies a complete machine learning workflow including:

- Dataset loading and cleaning
- Exploratory Data Analysis (EDA)
- Handling missing and invalid values
- Feature scaling
- Handling class imbalance using SMOTE
- Feature selection
- PCA visualization
- Model training
- Model evaluation and comparison

Two supervised learning algorithms were implemented:

1. **K-Nearest Neighbors (KNN)**
2. **Decision Tree Classifier**

Based on the evaluation results, the **Decision Tree Classifier** achieved the best overall performance.

---

## 🎯 Objective

The main objectives of this project are:

- To analyze water quality data using exploratory data analysis.
- To preprocess the dataset for machine learning.
- To handle class imbalance using SMOTE.
- To apply supervised learning models for water safety classification.
- To compare model performance using accuracy, precision, recall, F1-score, and confusion matrix.
- To identify the best-performing model for water quality classification.

---

## 🗂️ Dataset

The dataset used in this project is publicly available on Kaggle.

**Dataset Source:**  
[Water Quality Dataset - Kaggle](https://www.kaggle.com/datasets/mssmartypants/water-quality)

The dataset contains chemical and biological measurements of water samples.

### Dataset Summary

| Item | Description |
|---|---|
| Number of samples | Approximately 7,999 |
| Number of columns | 21 |
| Input features | 20 |
| Target variable | `is_safe` |
| Task type | Binary classification |

### Target Variable

| Label | Meaning |
|---|---|
| `0` | Not Safe |
| `1` | Safe |

---

## 🧪 Features

The dataset includes the following water quality parameters:

| Feature | Description |
|---|---|
| `aluminium` | Aluminium level in water |
| `ammonia` | Ammonia level in water |
| `arsenic` | Arsenic concentration |
| `barium` | Barium level |
| `cadmium` | Cadmium concentration |
| `chloramine` | Chloramine level |
| `chromium` | Chromium level |
| `copper` | Copper concentration |
| `flouride` | Fluoride level |
| `bacteria` | Bacterial contamination |
| `viruses` | Virus contamination |
| `lead` | Lead concentration |
| `nitrates` | Nitrate level |
| `nitrites` | Nitrite level |
| `mercury` | Mercury concentration |
| `perchlorate` | Perchlorate level |
| `radium` | Radium concentration |
| `selenium` | Selenium level |
| `silver` | Silver concentration |
| `uranium` | Uranium concentration |
| `is_safe` | Target label |

---

## ⚙️ Project Workflow

```text
Dataset Collection
        ↓
Data Cleaning
        ↓
Exploratory Data Analysis
        ↓
Data Preprocessing
        ↓
Feature Scaling
        ↓
Class Balancing using SMOTE
        ↓
Feature Selection
        ↓
PCA Visualization
        ↓
Model Training
        ↓
Model Evaluation
        ↓
Best Model Selection
```

---

## 📊 Exploratory Data Analysis

EDA was performed to understand the dataset and identify important patterns.

Main EDA steps included:

- Checking dataset shape and column types
- Checking missing and invalid values
- Analyzing class distribution
- Visualizing feature distributions
- Detecting outliers using boxplots
- Creating correlation heatmaps
- Comparing feature values between Safe and Not Safe classes

The dataset was highly imbalanced, with the majority of samples belonging to the **Not Safe** class.

---

## 🧹 Data Preprocessing

The following preprocessing steps were applied:

### 1. Handling Invalid Values

Invalid values such as `#NUM!` were converted into missing values and cleaned.

### 2. Handling Missing Values

Missing values were handled properly to ensure the dataset was ready for model training.

### 3. Feature Scaling

`StandardScaler` was used to scale the features.

Feature scaling was especially important for KNN because it is a distance-based algorithm.

### 4. Handling Class Imbalance

The dataset was highly imbalanced:

| Class | Meaning | Approximate Percentage |
|---|---|---|
| `0` | Not Safe | 88.59% |
| `1` | Safe | 11.41% |

To solve this issue, **SMOTE (Synthetic Minority Over-sampling Technique)** was applied to the training data.

### 5. Feature Selection

Feature selection was performed using statistical methods such as ANOVA F-test / SelectKBest.

Important features included:

- Aluminium
- Arsenic
- Silver
- Viruses
- Nitrates
- Bacteria
- Chloramine

---

## 🤖 Machine Learning Models

### 1. K-Nearest Neighbors

KNN is a distance-based classification algorithm. It predicts the class of a new sample by checking the nearest training samples.

### 2. Decision Tree Classifier

Decision Tree is a tree-based supervised learning algorithm. It splits the dataset based on feature values and creates decision rules for classification.

---

## 📈 Model Evaluation

The models were evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

### Model Performance Comparison

| Model | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| KNN | 0.7706 | 0.3085 | 0.8187 | 0.4481 |
| Decision Tree | **0.9450** | **0.7327** | 0.8132 | **0.7708** |

---

## 🏆 Best Model

The **Decision Tree Classifier** was selected as the best-performing model because it achieved:

- Highest accuracy
- Highest precision
- Highest F1-score
- Better overall balance between precision and recall
- Better interpretability through feature importance

Although KNN achieved slightly higher recall, Decision Tree performed better overall.

---

## 📌 Key Findings

- The dataset was highly imbalanced.
- SMOTE helped improve model learning for the minority class.
- Feature scaling improved the performance of KNN.
- Decision Tree achieved the best overall model performance.
- Important features such as aluminium, arsenic, silver, viruses, and nitrates played a major role in classification.
- Accuracy alone is not enough for imbalanced datasets; precision, recall, and F1-score are also important.

---

## 🛠️ Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- Imbalanced-learn
- Jupyter Notebook

---

## 📁 Repository Structure

```text
water-quality-classification-ml/
│
├── data/
│   └── README.md
│
├── notebooks/
│   └── Water_Quality_Classification.ipynb
│
├── src/
│   ├── preprocessing.py
│   ├── train.py
│   ├── evaluate.py
│   └── utils.py
│
├── reports/
│   ├── Report.pdf
│   └── figures/
│
├── models/
│   └── decision_tree_model.pkl
│
├── README.md
├── requirements.txt
├── .gitignore
├── LICENSE
└── CITATION.cff
```

---

## 🚀 Installation

Clone the repository:

```bash
git clone https://github.com/your-username/water-quality-classification-ml.git
cd water-quality-classification-ml
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate the environment:

### Windows

```bash
venv\Scripts\activate
```

### macOS/Linux

```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ How to Run

### Option 1: Run Jupyter Notebook

```bash
jupyter notebook
```

Then open:

```text
notebooks/Water_Quality_Classification.ipynb
```

### Option 2: Run Python Script

```bash
python src/train.py
python src/evaluate.py
```

---

## 📦 Requirements

Example `requirements.txt`:

```text
numpy
pandas
matplotlib
seaborn
scikit-learn
imbalanced-learn
jupyter
joblib
```

---

## 🔮 Future Work

Future improvements may include:

- Testing more models such as Random Forest, XGBoost, LightGBM, CatBoost, and SVM
- Applying cross-validation
- Adding hyperparameter tuning
- Using SHAP for explainable AI
- Building a Streamlit web application
- Deploying the model as an API
- Testing the model on real-world water quality data

---

## ⚠️ Limitations

- The dataset is publicly available and may not represent all real-world water conditions.
- SMOTE generates synthetic samples and may not fully capture real contamination patterns.
- Only two machine learning models were compared.
- External validation was not performed.
- This project should be used for educational purposes and not as a replacement for certified water testing.

---

## 👨‍💻 Author

**Md Towhidul Islam Tamim Mia**

Student ID: AIU24102441  
Albukhary International University  
School of Computing and Informatics

---

## 📜 License

This project is licensed under the MIT License.

---

## ⭐ Acknowledgement

The dataset was collected from Kaggle. This project was completed as part of a supervised machine learning assignment for water quality classification.
