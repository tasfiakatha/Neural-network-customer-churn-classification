import tensorflow as tf
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from tensorflow.keras.models import load_model

# Load the trained model, scaler and encoding pickle files
model = load_model('model.h5')

# Load label encoder
with open('label_encoder_gender.pkl','rb') as file:
    label_encoder_gender = pickle.load(file)

# Load one hot encoder
with open('onehot_encoder_geo.pkl','rb') as file:
    onehot_encoder_geo = pickle.load(file)

# Load standard encoder
with open('scaler.pkl','rb') as file:
    scaler = pickle.load(file)

# Streamlit app
st.title("Customer churn prediction")

# User input
geography = st.selectbox('Geography', onehot_encoder_geo.categories_[0])
gender = st.selectbox('Gender', label_encoder_gender.classes_)
age = st.slider('Age', 18, 100)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit score')
estimated_salary = st.number_input('Estimated salary')
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of products', 1, 4)
has_cr_card = st.selectbox('Has credit card', [0,1])
is_active_member = st.selectbox('Is active member', [0,1])

# Prepare input data
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

# Encode geography column
geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))

# Combine one hot encoded column with input data
input_df = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

# Scale input data
input_scaled = scaler.transform(input_df)

# Predict churn
prediction = model.predict(input_scaled)

# Select prediction probability
prediction_proba = prediction[0][0]

st.write(f"Churn probability: {prediction_proba:.2f}")

if prediction_proba > 0.5:
    st.subheader('Customer is likely to churn')
else:
    st.subheader('Customer is not likely to churn')