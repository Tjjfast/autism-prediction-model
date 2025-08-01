# Autism Spectrum Disorder (ASD) Prediction Project
This project aims to develop a machine learning model to predict the likelihood of Autism Spectrum Disorder in adults based on responses to the AQ-10 questionnaire and demographic data. The project includes a full data science workflow from data cleaning and exploratory analysis to model training, evaluation, and deployment as an interactive web application using Streamlit.

## 🌐 Live Demo
(Optional: Insert a link to your deployed Streamlit app here once it's live)
[Link to Live App]

## ✨ Features
- Data Cleaning & Preprocessing: Handles missing values, inconsistencies, and categorical data.

- Exploratory Data Analysis (EDA): Visualizes data distributions and correlations to uncover insights.

- Class Imbalance Handling: Implements the SMOTE (Synthetic Minority Over-sampling Technique) to address the imbalanced dataset.

- Model Training & Tuning: Trains and compares three different models (Decision Tree, Random Forest, XGBoost) and uses RandomizedSearchCV for hyperparameter optimization.

- Target Leakage Resolution: Identifies and removes a feature (result) that caused target leakage, ensuring a realistic and valid model.

- Interactive Web App: A user-friendly web interface built with Streamlit that allows users to input their data and get a real-time prediction.

## 🧪 Methodology
The project follows a standard machine learning workflow:

### 1. Data Source

This is the [dataset](https://www.kaggle.com/datasets/shivamshinde123/autismprediction) used.

### 2. Data Cleaning and EDA

- Handling Missing Values: Missing data, represented by '?', was identified and imputed (e.g., replaced with the mode 'Others' for ethnicity).

- Feature Consolidation: Categorical features with high cardinality, like relation, were simplified to reduce complexity.

- Outlier Treatment: Outliers in numerical columns (age, result) were identified using the IQR method and replaced with the median value to prevent them from skewing the model.

- EDA Findings: The primary finding was a significant class imbalance, with the non-autistic class representing approximately 80% of the data. This highlighted that accuracy would not be a sufficient evaluation metric.

### 3. Target Leakage
Heatmap revealed a perfect correlation (1.00) between the result feature and the target Class/ASD. This feature was identified as a target leak and was removed from the dataset before training to ensure the model's predictive power on new, unseen data.

### 4. Modeling and Evaluation
- Data Splitting: The data was split into training (80%) and testing (20%) sets.

- Handling Imbalance: SMOTE was applied only to the training data to create synthetic samples of the minority class (ASD positive), balancing the class distribution for the model to learn from.

- Model Selection: Three models were evaluated using 5-fold cross-validation:

  - Decision Tree Classifier

  - Random Forest Classifier

  - XGBoost Classifier

- Hyperparameter Tuning: RandomizedSearchCV was used to find the optimal parameters for each model. The Random Forest Classifier emerged as the best-performing model.

- Final Evaluation: The tuned Random Forest model was evaluated on the unseen test set.

## 📈 Results
The final model achieved an overall accuracy of 82% on the test set.

Recall for ASD (Class 1) is 64%: This is the most critical metric for this use case. It means the model correctly identifies 64% of all actual ASD-positive cases. While a good start, it also means 36% are missed (false negatives).

Precision for ASD (Class 1) is 59%: When the model predicts a user has ASD, it is correct 59% of the time.

## 🧰 How to Run This Project Locally
### Prerequisites
- Python 3.8+
### Instructions
Clone the repository:
```
git clone https://github.com/Tjjfast/autism-prediction-model.git
cd [your-repo-folder]
```
Create and activate a virtual environment:

For Windows
```
python -m venv venv
venv\Scripts\activate
```
For macOS/Linux
```
python3 -m venv venv
source venv/bin/activate
```
Install the required libraries:
```
pip install -r requirements.txt
```
Run the Streamlit app:
```
streamlit run app.py
```
The application will open in your web browser.

### 🛠️ Technologies Used
- 🐍 Python

- 🧮 Pandas & NumPy for data manipulation

- 🧠 Scikit-learn for modeling and preprocessing

- ⚡ XGBoost for modeling

- 📊 Matplotlib & Seaborn for data visualization

- ⚖️ Imbalanced-learn for SMOTE

- 🌐 Streamlit for the interactive web app

- 📓 Jupyter Notebook for experimentation
