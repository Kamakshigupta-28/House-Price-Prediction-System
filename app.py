import streamlit as st
import pickle
import numpy as np
import pandas as pd

# -----------------------------
# Page Configuration
# -----------------------------

st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="wide"
)

# -----------------------------
# Load Model
# -----------------------------

with open("house_price_model.pkl", "rb") as file:
    model = pickle.load(file)

# -----------------------------
# Title
# -----------------------------

st.title("🏠 House Price Prediction System")

st.write(
    "This application predicts the price of a house using "
    "Machine Learning and Multiple Linear Regression."
)

st.divider()

# -----------------------------
# Sidebar
# -----------------------------

st.sidebar.header("🏡 Enter House Details")

area = st.sidebar.number_input(
    "Area (sq. ft.)",
    min_value=300,
    max_value=10000,
    value=1500,
    step=100
)

bedrooms = st.sidebar.number_input(
    "Number of Bedrooms",
    min_value=1,
    max_value=10,
    value=3,
    step=1
)

bathrooms = st.sidebar.number_input(
    "Number of Bathrooms",
    min_value=1,
    max_value=10,
    value=2,
    step=1
)

age = st.sidebar.number_input(
    "House Age (Years)",
    min_value=0,
    max_value=100,
    value=5,
    step=1
)

distance_center = st.sidebar.number_input(
    "Distance from City Center (km)",
    min_value=0.0,
    max_value=100.0,
    value=5.0,
    step=0.5
)

# -----------------------------
# Prediction Button
# -----------------------------

if st.sidebar.button("🔮 Predict House Price"):

    input_data = np.array([
        [
            area,
            bedrooms,
            bathrooms,
            age,
            distance_center
        ]
    ])

    prediction = model.predict(input_data)

    predicted_price = prediction[0]

    st.success("Prediction Completed Successfully!")

    st.subheader("💰 Predicted House Price")

    st.metric(
        label="Estimated Price",
        value=f"₹ {predicted_price:,.2f}"
    )

    st.info(
        f"The estimated price of this house is "
        f"₹ {predicted_price:,.2f}"
    )

# -----------------------------
# Input Summary
# -----------------------------

st.divider()

st.subheader("📊 House Information")

input_df = pd.DataFrame({
    "Feature": [
        "Area",
        "Bedrooms",
        "Bathrooms",
        "House Age",
        "Distance from Center"
    ],
    "Value": [
        f"{area} sq. ft.",
        bedrooms,
        bathrooms,
        f"{age} years",
        f"{distance_center} km"
    ]
})

st.table(input_df)

# -----------------------------
# About Model
# -----------------------------

st.divider()

st.subheader("🤖 About the Machine Learning Model")

st.write("""
The system uses Multiple Linear Regression.

The model learns the relationship between:

- Area
- Number of Bedrooms
- Number of Bathrooms
- House Age
- Distance from City Center

and predicts the house price.
""")

st.latex(
    "Price = b_0 + b_1(Area) + b_2(Bedrooms) + "
    "b_3(Bathrooms) + b_4(Age) + b_5(Distance)"
)