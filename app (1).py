import streamlit as st
import joblib
import pandas as pd

# Load model and imputer
model = joblib.load('availability_model.pkl')
imputer = joblib.load('imputer.pkl')

st.title("🏡 Room Availability Checker")

bedrooms = st.number_input("Bedrooms", 0, 10, 1)
bathrooms = st.number_input("Bathrooms", 0, 10, 1)
price = st.number_input("Price (KES)", 0.0, 100000.0, 1000.0)
rate = st.number_input("Rate (optional)", 0.0, 5.0, 0.0)

if st.button("Check Availability"):
    input_df = pd.DataFrame([{
        'bedrooms': bedrooms,
        'bathrooms': bathrooms,
        'price': price,
        'rate': rate
    }])
    input_df = imputer.transform(input_df)
    prediction = model.predict(input_df)[0]
    result = "✅ Available!" if prediction == 1 else "❌ Not Available"
    st.success(result) if prediction == 1 else st.error(result)
