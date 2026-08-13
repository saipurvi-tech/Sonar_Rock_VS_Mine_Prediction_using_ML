# Sonar_Rock_VS_Mine_Prediction_using_ML

> **Detect underwater hazards before it's too late!**  
> A machine learning model that analyzes sonar return signals to classify objects as either a harmless **Rock (R)** or a dangerous naval **Mine (M)**.

---

## 📌 Project Overview

Submarines use sonar technology to navigate the ocean floor, but distinguishing between natural rock formations and underwater explosive mines can be tricky. This project leverages **Logistic Regression** to analyze multi-band sonar signals and accurately predict whether a sonar return signal is reflected from a **Rock** or a **Mine**.

---

## 🛠️ Tech Stack & Dependencies

* **Language:** Python 
* **Data Processing:** `pandas`, `numpy`
* **Machine Learning:** `scikit-learn` (Logistic Regression, Train-Test Split, Accuracy Metrics)

---

## 📊 Dataset Breakdown

* **Total Samples:** 208 sonar returns
* **Features:** 60 numerical sonar signal strength values (frequencies)
* **Classes:** 
  * `M` ➔ Naval Mine (111 instances)
  * `R` ➔ Underwater Rock (97 instances)

---

## 🎯 Model Performance

* **Training Accuracy:** `~83.4%`
* **Testing Accuracy:** `~76.2%`

---

---

## 🧠 What I Learned

Building this project provided hands-on experience with end-to-end Machine Learning workflows:

* **Data Exploration & Preprocessing:**
  * Handling datasets without pre-existing column headers (`header=None`).
  * Utilizing Pandas for statistical analysis using `.describe()`, dataset profiling using `.shape`, and class distribution checks using `.value_counts()`.

* **Feature & Target Separation:**
  * Slicing continuous feature columns ($X$) away from categorical label targets ($Y$).

* **Model Training & Evaluation:**
  * Implementing **Stratified Train-Test Splitting** (`stratify=Y`) to ensure both training and testing datasets maintain a balanced ratio of Rocks and Mines.
  * Building a binary classification model using **Logistic Regression**.
  * Evaluating model generalization performance by comparing **Training Accuracy (~83.4%)** against **Test Accuracy (~76.2%)** to monitor for potential overfitting.

* **Deployment-Ready Inference:**
  * Reshaping $1\text{D}$ input arrays into $2\text{D}$ instances (`reshape(1, -1)`) so the trained model can make predictions on new, single sonar data readings.
