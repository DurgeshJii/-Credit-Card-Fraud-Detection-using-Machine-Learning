# Credit-Card-Fraud-Detection-using-Machine-Learning
His project focuses on building a machine learning model to detect fraudulent credit card transactions using a highly imbalanced dataset.

# 📊 Dataset Overview
Total Transactions: 100,000+
Features:
TransactionID
TransactionDate
Amount
MerchantID
TransactionType
Location
Target: IsFraud (0 = Normal, 1 = Fraud)

# ⚠️ Key Challenge

The dataset is highly imbalanced:

99% Normal Transactions
1% Fraud Transactions
⚙️ Project Workflow
🔹 1. Data Preprocessing
Converted datetime features
Extracted Year, Month, Day
Dropped unnecessary columns
🔹 2. Feature Engineering
One-hot encoding for categorical variables
Created structured input for ML models
🔹 3. Handling Imbalance
Applied SMOTE to balance training data
Improved model learning for minority class
🔹 4. Model Building
Logistic Regression
Random Forest Classifier
🔹 5. Model Evaluation

# Used:

Accuracy
Confusion Matrix
Precision, Recall, F1-score

# 📈 Results:

Logistic Regression Accuracy: ~97%
Random Forest Accuracy: ~98%

⚠️ Note: Accuracy is not sufficient → Focus on fraud detection metrics

🔹 6. Feature Importance

Identified top contributing features like:

TransactionID
Month
Amount
MerchantID

# 🛠️ Tech Stack
Python
Pandas, NumPy
Scikit-learn
Imbalanced-learn (SMOTE)
Matplotlib

# 🚀 How to Run
git clone <repo-link>
cd fraud-detection
pip install -r requirements.txt
jupyter notebook

# 🔮 Future Enhancements
XGBoost / LightGBM models
Hyperparameter tuning
Real-time fraud detection API
Model deployment

# 📌 Conclusion

This project demonstrates how to handle real-world imbalanced datasets and build effective fraud detection systems using machine learning techniques.

# Durgesh Yadav
