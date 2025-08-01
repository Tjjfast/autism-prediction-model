import streamlit as st
import pandas as pd
import pickle
import numpy as np

# --- LOAD THE SAVED FILES ---

# Load the trained model
try:
    with open('autism_model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
except FileNotFoundError:
    st.error("Model file not found. Please ensure 'best_model.pkl' is in the correct directory.")
    st.stop()
except Exception as e:
    st.error(f"Error loading the model: {e}")
    st.stop()

# Load the label encoders
try:
    with open('encoders.pkl', 'rb') as encoder_file:
        encoders = pickle.load(encoder_file)
except FileNotFoundError:
    st.error("Encoder file not found. Please ensure 'encoders.pkl' is in the correct directory.")
    st.stop()
except Exception as e:
    st.error(f"Error loading the encoders: {e}")
    st.stop()
    
# --- APP TITLE AND DESCRIPTION ---

st.title("Autism Spectrum Disorder (ASD) Detector 🧠")
st.write(
    "This web app uses a machine learning model to predict the likelihood of Autism "
    "Spectrum Disorder based on the AQ-10 questionnaire scores and demographic data. "
    "Please fill in the details below."
)
st.warning("**Disclaimer:** This is a portfolio project and not a medical diagnostic tool. Please consult a healthcare professional for an actual diagnosis.")

# --- CREATE THE USER INPUT INTERFACE ---

st.header("Please answer the following questions:")

# Create two columns for a better layout
col1, col2 = st.columns(2)

# AQ-10 Questions (Based on A1_Score to A10_Score)
with col1:
    a1 = st.selectbox("1. I often notice small sounds when others do not.", ["No", "Yes"])
    a2 = st.selectbox("2. I usually concentrate more on the whole picture, rather than the small details.", ["No", "Yes"])
    a3 = st.selectbox("3. I find it easy to do more than one thing at once.", ["No", "Yes"])
    a4 = st.selectbox("4. If there is an interruption, I can switch back to what I was doing very quickly.", ["No", "Yes"])
    a5 = st.selectbox("5. I find it easy to ‘read between the lines’ when someone is talking to me.", ["No", "Yes"])
    
with col2:
    a6 = st.selectbox("6. I know how to tell if someone listening to me is getting bored.", ["No", "Yes"])
    a7 = st.selectbox("7. When I’m reading a story, I find it difficult to work out the characters’ intentions.", ["No", "Yes"])
    a8 = st.selectbox("8. I like to collect information about categories of things (e.g., types of cars, birds, trains).", ["No", "Yes"])
    a9 = st.selectbox("9. I find it difficult to understand what other people are feeling.", ["No", "Yes"])
    a10 = st.selectbox("10. I find it very easy to play games with children that involve pretending.", ["No", "Yes"])

st.header("Demographic Information:")

# Demographic inputs
age = st.number_input("Age (in years)", min_value=2, max_value=80, value=25)
gender = st.selectbox("Gender", options=['f', 'm']) # Options from your encoder
ethnicity = st.selectbox("Ethnicity", options=['Others', 'White-European', 'Middle Eastern ', 'Pasifika', 'Black', 'Hispanic', 'Asian', 'Turkish', 'South Asian', 'Latino'])
jaundice = st.selectbox("Was the user born with jaundice?", options=['no', 'yes'])
austim = st.selectbox("Does the user have a family member with ASD?", options=['no', 'yes'])
contry_of_res = st.selectbox("Country of Residence", options=['Austria', 'India', 'United States', 'South Africa', 'Jordan', 'United Kingdom', 'Brazil', 'New Zealand', 'Canada', 'Kazakhstan', 'United Arab Emirates', 'Australia', 'Ukraine', 'Iraq', 'France', 'Malaysia', 'Vietnam', 'Egypt', 'Netherlands', 'Afghanistan', 'Oman', 'Italy', 'Bahamas', 'Saudi Arabia', 'Ireland', 'Aruba', 'Sri Lanka', 'Russia', 'Bolivia', 'Azerbaijan', 'Armenia', 'Serbia', 'Ethiopia', 'Sweden', 'Iceland', 'China', 'Angola', 'Germany', 'Spain', 'Tonga', 'Pakistan', 'Iran', 'Argentina', 'Japan', 'Mexico', 'Nicaragua', 'Sierra Leone', 'Czech Republic', 'Niger', 'Romania', 'Cyprus', 'Belgium', 'Burundi', 'Bangladesh'])
used_app_before = st.selectbox("Has the user used this type of app before?", options=['no', 'yes'])
relation = st.selectbox("Relation to the user", options=['Self', 'Others'])


# --- PREDICTION LOGIC ---

if st.button("Get Prediction", key='predict_button'):
    # Map Yes/No answers to 1/0
    # Note: The mapping logic here is a simplification. The actual AQ-10 scoring is nuanced.
    # For a real application, you would implement the correct scoring rules.
    # Based on your notebook, the model expects 1s and 0s directly.
    a_scores = [1 if ans == "Yes" else 0 for ans in [a1, a2, a3, a4, a5, a6, a7, a8, a9, a10]]

    # Create a dictionary for the input data
    input_data = {
        'A1_Score': [a_scores[0]], 'A2_Score': [a_scores[1]], 'A3_Score': [a_scores[2]],
        'A4_Score': [a_scores[3]], 'A5_Score': [a_scores[4]], 'A6_Score': [a_scores[5]],
        'A7_Score': [a_scores[6]], 'A8_Score': [a_scores[7]], 'A9_Score': [a_scores[8]],
        'A10_Score': [a_scores[9]],
        'age': [age],
        'gender': [gender],
        'ethnicity': [ethnicity],
        'jaundice': [jaundice],
        'austim': [austim], # Note: 'austim' is the column name in your dataset
        'contry_of_res': [contry_of_res],
        'used_app_before': [used_app_before],
        'relation': [relation]
    }
    
    # Create a DataFrame from the input
    input_df = pd.DataFrame(input_data)
    
    # Encode the categorical features using the loaded encoders
    for column, encoder in encoders.items():
        # Check if the column exists in the input dataframe to avoid errors
        if column in input_df.columns:
            # Use a try-except block to handle unseen labels gracefully
            try:
                input_df[column] = encoder.transform(input_df[column])
            except ValueError:
                st.error(f"The value for '{column}' was not seen during training. Please select a different value.")
                # Handle this case, e.g., by stopping or assigning a default encoded value if appropriate
                st.stop()
    
    # IMPORTANT: The 'result' column was identified as a target leak and should not be in the input.
    # Make sure your final model was trained without it. If it was, add it here for the prediction.
    # For this example, I am assuming the final model was correctly retrained without 'result'.

    # Make prediction
    prediction = model.predict(input_df)
    prediction_proba = model.predict_proba(input_df)
    
    # --- DISPLAY THE RESULT ---
    
    st.header("Prediction Result")
    
    if prediction[0] == 1:
        st.warning("The model predicts that the user is **potentially on the Autism Spectrum**.")
        st.write(f"Confidence Score: {prediction_proba[0][1]*100:.2f}%")
    else:
        st.success("The model predicts that the user is **likely not on the Autism Spectrum**.")
        st.write(f"Confidence Score: {prediction_proba[0][0]*100:.2f}%")
        
    st.info("Please remember this is a prediction based on a model and not a medical diagnosis.")