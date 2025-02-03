# Machine-Learning: MultiClass-MultiLabel-SVM-KMeans
# 🔷 Multi-Class and Multi-Label Classification with Support Vector Machines and K-Means Clustering

## 🔶 Overview
This project explores **multi-class and multi-label classification** using **Support Vector Machines (SVMs)** and **unsupervised learning with K-Means Clustering**. The dataset used is the **Anuran Calls (MFCCs) dataset**, which involves classifying frog calls into different taxonomic categories. We evaluate **binary relevance**, **classifier chaining**, and **clustering-based label prediction**, optimizing performance with **cross-validation** and **SMOTE for class imbalance**.

## 🔷 Datasets Used
### **UCI Machine Learning Repository - Anuran Calls (MFCCs)**
**Dataset:** [Anuran Calls (MFCCs)](https://archive.ics.uci.edu/ml/datasets/Anuran+Calls+%28MFCCs%29)
- **Multi-label dataset** containing frog calls labeled with **Family, Genus, and Species**.
- Each data point is assigned **three labels**, making it suitable for **multi-label classification**.
- Used for both **supervised classification (SVMs)** and **unsupervised clustering (K-Means)**.

## 🔷 Libraries Used
- **pandas** - Data manipulation and preprocessing.
- **numpy** - Numerical computations.
- **matplotlib & seaborn** - Data visualization.
- **sklearn.svm** - Support Vector Machines (SVMs).
- **sklearn.cluster** - K-Means clustering.
- **imblearn** - SMOTE for handling class imbalance.
- **scipy** - Statistical analysis.

## 🔷 Steps Taken to Accomplish the Project

### 🔶 1. Multi-Class & Multi-Label Classification with Support Vector Machines
- Downloaded and preprocessed the **Anuran Calls (MFCCs) dataset**.
- Splitted **70% data for training** and used the rest for evaluation.
- Implemented **binary relevance** by training **one SVM per label** (`Family`, `Genus`, `Species`).
- Evaluated performance using **Exact Match Score, Hamming Score, and Hamming Loss**.

### 🔶 2. Hyperparameter Tuning for SVM
- Trained **Gaussian kernel SVMs** using a **one-vs-all** approach.
- Used **10-fold cross-validation** to determine:
  - Optimal **penalty parameter (λ)**
  - Best **Gaussian kernel width (σ)**
- Compared results between **standardized and raw features**.

### 🔶 3. L1-Penalized SVM for Feature Selection
- Applied **L1-regularization** to encourage **sparse feature selection**.
- Standardized attributes and optimized **λ** via **cross-validation**.
- Identified key **MFCC features** relevant to frog call classification.

### 🔶 4. Addressing Class Imbalance with SMOTE
- Used **Synthetic Minority Over-sampling (SMOTE)** to balance training data.
- Retrained the **L1-SVM** on balanced data.
- Analyzed improvements in **recall and F1-score**.

### 🔶 5. K-Means Clustering for Label Inference
- Applied **unsupervised K-Means clustering** (without train-test split).
- Selected **optimal k** (`1 ≤ k ≤ 50`) using:
  - **CH Statistics**
  - **Gap Statistics**
  - **Silhouette Score**
- Assigned **majority labels** to each cluster.

### 🔶 6. Evaluating Clustering Performance
- Compared **predicted cluster labels** to actual labels (`Family, Genus, Species`).
- Measured **Hamming Distance, Hamming Score, and Hamming Loss**.
- Reported **mean and standard deviation** over **50 Monte Carlo iterations**.

### 🔶 7. Extra Practice & Extensions
- Explored **Classifier Chains** for multi-label classification.
- Computed **Precision, Recall, ROC, and AUC** for multi-label SVM models.

---
## 📌 **Note**
This repository contains a **Jupyter Notebook** detailing each step, along with **results and visualizations**.
