
Credit Card Approval Prediction 🏦📊

This project aims to predict whether a credit card application will be approved using machine learning techniques. It utilizes the UCI Credit Approval dataset and applies data preprocessing, SMOTE for class imbalance, and ensemble models like Random Forest and Gradient Boosting for accurate predictions.

📁 Dataset
Source: UCI Machine Learning Repository – Credit Approval Data

The dataset contains both categorical and numerical attributes with some missing values.

The target variable (+ or -) indicates whether the credit card application was approved.

⚙️ Technologies Used
Python

pandas, NumPy

scikit-learn

imbalanced-learn (SMOTE)

seaborn, matplotlib

🚀 Workflow
Data Loading and Exploration

Read dataset and assign column names.

Basic statistical and null value analysis.

Preprocessing

Handle missing values (using median and forward fill).

Label encoding of categorical features.

Feature scaling using StandardScaler.

Handling Imbalance

Applied SMOTE (Synthetic Minority Oversampling Technique) to balance target classes.

Modeling

Trained and evaluated:

Random Forest

Gradient Boosting

Used accuracy, classification report, and confusion matrix for evaluation.

Prediction

Implemented a function to predict approval based on new input data.

(Optional) Fairness Analysis

Checked for demographic bias using one of the features (A1).

📊 Results
Achieved ~92% accuracy with both models.

Models showed balanced precision and recall.

Sample predictions included various applicant scenarios.

🔍 Sample Prediction Function

def predict_credit_approval(input_data, model):
    input_df = pd.DataFrame([input_data], columns=data.columns[:-1])
    input_scaled = pd.DataFrame(scaler.transform(input_df), columns=data.columns[:-1])
    prediction = model.predict(input_scaled)
    return "Approved" if prediction[0] == 1 else "Rejected"
🧠 Future Enhancements
Deploy the model using Streamlit or Flask.

Implement a proper fairness metric (e.g., disparate impact).

Tune hyperparameters for model optimization.

📌 How to Run
Clone this repo:


git clone https://github.com/your-username/credit-card-approval.git
cd credit-card-approval
Install dependencies:


pip install -r requirements.txt
Run the script:


python credit_approval_model.py
📜 License
This project is for educational purposes only and uses open data from the UCI repository.

