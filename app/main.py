"""
    Frontend app built with Streamlit
"""

import streamlit as st
from app_utils import regression_predict, classification_predict, preprocess, load_dataset, print_info

st.set_page_config(page_title="California Housing Input", page_icon="🏠")

st.title("California Housing Price Prediction")
st.write("Enter the house information below.")

longitude = st.number_input("Longitude", value=-122.23, format="%.6f")
latitude = st.number_input("Latitude", value=37.88, format="%.6f")
housing_median_age = st.number_input("Housing Median Age", min_value=1, value=20)
total_rooms = st.number_input("Total Rooms", min_value=1, value=1000)
total_bedrooms = st.number_input("Total Bedrooms", min_value=1, value=200)
population = st.number_input("Population", min_value=1, value=800)
households = st.number_input("Households", min_value=1, value=300)
median_income = st.number_input("Median Income", min_value=0.0, value=3.5, format="%.4f")
ocean_proximity = st.selectbox(
    "Ocean Proximity",
    [
        "NEAR BAY",
        "<1H OCEAN",
        "INLAND",
        "NEAR OCEAN",
        "ISLAND",
    ]
)

if st.button("Submit"):
    data = preprocess(longitude, latitude, housing_median_age, total_rooms, total_bedrooms, population, households, median_income, ocean_proximity )
    X, y, y_classification= load_dataset()
    regression = regression_predict(data, X, y)
    log_regression, knn, lda, qda, naive_bayes = classification_predict(data, X, y_classification)

    print_info(regression, log_regression, knn, lda, qda, naive_bayes)