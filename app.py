import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="House Price Predictor", page_icon="🏠")


@st.cache_resource                       # load the model once, not on every click
def load_model():
    return joblib.load("house_price_model.sav")     # relative path so it works on Streamlit Cloud


model = load_model()

# Ranges taken from the cleaned training data (used for sliders and warnings)
RANGES = {
    "Area_m2": (26.0, 426.4),
    "House_Age_Years": (0.4, 75.0),
    "Distance_to_City_km": (0.07, 27.4),
}
NEIGHBORHOODS = ["Gasabo", "Huye", "Kicukiro", "Kigali City", "Musanze", "Nyarugenge"]

st.title("🏠 House Price Predictor")
st.write(
    "Enter the details of a house in Rwanda and this app estimates its market price "
    "in million RWF using a multiple linear regression model trained on past sales."
)

col1, col2 = st.columns(2)
with col1:
    area = st.number_input("Area (m²)", min_value=10.0, max_value=1000.0, value=105.0, step=1.0)
    bedrooms = st.number_input("Bedrooms", min_value=1, max_value=10, value=3, step=1)
    bathrooms = st.number_input("Bathrooms", min_value=1, max_value=10, value=3, step=1)
    parking = st.number_input("Parking spaces", min_value=0, max_value=6, value=1, step=1)
with col2:
    age = st.number_input("House age (years)", min_value=0.0, max_value=100.0, value=7.0, step=0.5)
    distance = st.number_input("Distance to city centre (km)", min_value=0.0, max_value=100.0,
                               value=3.0, step=0.5)
    neighborhood = st.selectbox("Neighborhood", NEIGHBORHOODS, index=0)

if st.button("Predict price", type="primary"):
    # Warn (do not block) when inputs fall outside what the model saw in training
    checks = {"Area_m2": area, "House_Age_Years": age, "Distance_to_City_km": distance}
    for name, value in checks.items():
        low, high = RANGES[name]
        if not (low <= value <= high):
            st.warning(f"{name} = {value} is outside the training range ({low} to {high}). "
                       "The prediction may be unreliable.")
    if bedrooms > bathrooms + 3:
        st.info("Many more bedrooms than bathrooms is unusual in the training data.")

    # One-row DataFrame with the exact column names used in training
    row = pd.DataFrame([{
        "Area_m2": area,
        "Bedrooms": bedrooms,
        "Bathrooms": bathrooms,
        "House_Age_Years": age,
        "Distance_to_City_km": distance,
        "Parking_Spaces": parking,
        "Neighborhood": neighborhood,
    }])
    price = model.predict(row)[0]
    st.success(f"Estimated price: **{price:,.1f} million RWF**")
    st.caption("Typical error on unseen houses is about ±23 million RWF (test RMSE).")
               
               
               