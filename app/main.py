"""
    Frontend app built with Streamlit
"""

import streamlit as st
from app_utils import regression_predict, classification_predict, preprocess, load_dataset, print_info

st.set_page_config(page_title="California Housing", page_icon="🏠")

st.title("California Housing Price Prediction")
st.write("Enter the district's information below")

longitude =st.number_input("Longitude", value=-119)
latitude = st.number_input("Latitude", value=35)
housing_median_age = st.number_input("Housing Median Age", min_value=1, value=28)
total_rooms = st.number_input("Total Rooms", min_value=1, value=2600)
total_bedrooms = st.number_input("Total Bedrooms", min_value=1, value=500)
population = st.number_input("Population", min_value=1, value=1500)
households =st.number_input("Households", min_value=1, value=500)
median_income= st.number_input("Median Income (in $)", min_value=0.0, value=3000.0) / 1000
ocean_proximity = st.selectbox(
    "Ocean Proximity",["NEAR BAY", "<1H OCEAN", "INLAND","NEAR OCEAN","ISLAND"]
)
if st.button("Predict"):
    data = preprocess(longitude, latitude, housing_median_age, total_rooms, total_bedrooms, population, households, median_income, ocean_proximity )
    X, y, y_classification= load_dataset()
    regression = regression_predict(data, X, y)
    log_regression, knn, lda, qda, naive_bayes = classification_predict(data, X, y_classification)

    print_info(regression, log_regression, knn, lda, qda, naive_bayes)