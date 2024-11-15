import streamlit as st
import pandas as pd
import pickle
from PIL import Image

# Load the pre-trained model and preprocessor
with open('best_rf_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('preprocessor.pkl', 'rb') as preprocessor_file:
    preprocessor = pickle.load(preprocessor_file)

    body_type_images = {
    'Sedan': 'data/sedan.jpg',
    'SUV': 'data/suv.jpg',
    'Hatchback': 'data/hatchback.jpg',
    'Convertibles': 'data/convert.jpg',
}

st.title("Car Price Prediction App")
st.sidebar.header("Enter Car Details")

# User input fields
fuel_type = st.sidebar.selectbox("Fuel Type", ["Petrol", "Diesel", "Electric",'Lpg', 'Cng'])
body_type = st.sidebar.selectbox("Body Type", list(body_type_images.keys()))
if body_type in body_type_images:
    image_path = body_type_images[body_type]
    image = Image.open(image_path)
    st.image(image, caption=f"{body_type}", use_container_width=True)

transmission = st.sidebar.selectbox("Transmission", ["Manual", "Automatic"])
km_driven = st.sidebar.number_input("Kilometers Driven", min_value=0, value=10000, step=1000)
model_year = st.sidebar.number_input("Model Year", min_value=2001, max_value=2024, value=2019)
engine_size = st.sidebar.number_input("Engine (CC)", min_value=1000, max_value=6000, value=2000)
brand = st.sidebar.selectbox("Brand", ['Maruti' ,'Ford' ,'Tata' ,'Hyundai', 'Jeep' ,'Datsun' ,'Honda' ,'Mahindra'
 ,'Mercedes-Benz' ,'BMW' ,'Renault' ,'Audi' ,'Toyota', 'Mini', 'Kia', 'Skoda'
 ,'Volkswagen' ,'Volvo' ,'MG' ,'Nissan' ,'Fiat' ,'Mitsubishi' ,'Jaguar' ,'Land Rover' ,'Chevrolet' ,'Citroen' , 'Isuzu', 'Lexus', 'Porsche'])
city = st.sidebar.selectbox("City", ["bangalore", "chennai", "delhi", "hyderabad", "jaipur", "kolkata"])
registration_year = st.sidebar.number_input("Registration Year", min_value=1990, max_value=2023, value=2015)

# Button to make a prediction
if st.sidebar.button("Predict Price"):
    # Prepare data for prediction
    data = pd.DataFrame({
        'Fuel_Type': [fuel_type],
        'Body_Type': [body_type],
        'Transmission': [transmission],
        'KM_Driven': [km_driven],
        'Model_Year': [model_year],
        'Engine': [engine_size],
        'Brand': [brand],
        'City': [city],
        'Registration_Year': [registration_year]
    })

    data_preprocessed = preprocessor.transform(data)
    price_prediction = model.predict(data_preprocessed)
    st.write(f"Predicted Price of the Car: ₹ {price_prediction[0]:.2f} Lakh")


