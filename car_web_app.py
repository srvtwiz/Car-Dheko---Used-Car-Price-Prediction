import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Load the saved model and preprocessor
with open('best_rf_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Define categorical and numerical columns
categorical_cols = ['Fuel_Type', 'Body_Type', 'Transmission', 'Brand', 'City']
numeric_cols = ['KM_Driven', 'Engine', 'Registration_Year', 'Model_Year']

# Create transformations
preprocessed=ColumnTransformer(transformers=[('num',StandardScaler(),numeric_cols),('cat',OneHotEncoder(),categorical_cols)])

# Streamlit App Layout
st.title("Car Price Prediction")
st.write("This application predicts the price of a used car based on various features.")

# Input Fields for each feature
fuel_type = st.selectbox('Fuel Type', ['Petrol', 'Diesel', 'Electric', 'CNG'])
body_type = st.selectbox('Body Type', ['SUV', 'Sedan', 'Hatchback', 'Convertible', 'Coupe', 'Wagon'])
transmission = st.selectbox('Transmission', ['Manual', 'Automatic'])
brand = st.selectbox('Brand', ['Toyota', 'Honda', 'Ford', 'BMW', 'Mercedes', 'Audi'])  # Adjust brand list accordingly
# model_name = st.text_input('Model')
city = st.selectbox('City', ['Bangalore', 'Chennai', 'Delhi', 'Hyderabad', 'Jaipur', 'Kolkata'])
km_driven = st.number_input('Kilometers Driven', min_value=0)
engine_size = st.number_input('Engine Size (in CC)', min_value=0)
registration_year = st.number_input('Registration Year', min_value=1900, max_value=2024, value=2015)
model_year = st.number_input('Model Year', min_value=1900, max_value=2024, value=2015)

# Create DataFrame with user input
input_data = pd.DataFrame({
    'Fuel_Type': [fuel_type],
    'Body_Type': [body_type],
    'Transmission': [transmission],
    'Brand': [brand],
    # 'Model': [model_name],
    'City': [city],
    'KM_Driven': [km_driven],
    'Engine': [engine_size],
    'Registration_Year': [registration_year],
    'Model_Year': [model_year]
})

# Preprocess the input data
input_data_transformed = preprocessed.fit_transform(input_data)

# Predict the price
predicted_price = model.predict(input_data_transformed)

# Display the predicted price
st.write(f"The estimated price of the car is: ₹ {predicted_price[0]:,.2f} Lakh")
