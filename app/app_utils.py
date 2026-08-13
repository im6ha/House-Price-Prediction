import pickle
import numpy as np
import pandas as pd
import streamlit as st

from src.regression import KNN_prediction
from src.classification import logistic_function, K_nearest_neighbors, discriminant_analysis_one_point,naive_bayes_probabilities

def preprocess(longitude, latitude, housing_median_age, total_rooms, total_bedrooms, population, households, median_income, ocean_proximity ):
    """
    Preprocess the user input (standrize and log-transform)
    """
    #one hot encoding
    lt_1h_ocean, inland, near_ocean, island = 0,0,0,0
    if ocean_proximity =="<1H OCEAN":
        lt_1h_ocean=1
    elif ocean_proximity == "INLAND":
        inland=1
    elif ocean_proximity== "ISLAND":
        island = 1
    elif ocean_proximity =="NEAR OCEAN":
        near_ocean = 1

    #adding the engineered features
    rooms_per_household = total_rooms/households
    bedrooms_per_room = total_bedrooms / total_rooms
    population_per_household = population / households

    data = {
        'longitude':longitude,
        'latitude' : latitude,
        'housing_median_age': housing_median_age,
        'bedrooms_per_room':bedrooms_per_room,
        'population_per_household': population_per_household,
        'rooms_per_household':rooms_per_household,
        'log_total_rooms':total_rooms,
        'log_total_bedrooms':total_bedrooms,
        'log_population':population,
        'log_households':households,
        'log_median_income':median_income,
        '<1h ocean':lt_1h_ocean,
        'inland':inland,
        'near ocean':near_ocean,
        'island':island
    }

    with open('../cache/stats.pkl', 'rb') as f :
        stats = pickle.load(f)

    #log transforms
    for col in stats["log_transformed_cols"]:
        data['log_'+col] = np.log1p(data['log_'+col])

    #standrize
    for col, mean_std in stats["means_stds"].items():
        mean = mean_std[0]
        std= mean_std[1]
        data[col] = (data[col] - mean)/std

    return data

def load_dataset():
    """
    Load the dataset as X and y (for regression and classification)
    """
    training_dataset = pd.read_csv('../data/processed/training_data.csv')
    testing_dataset = pd.read_csv('../data/processed/testing_data.csv')
    complete_dataset = pd.concat([training_dataset, testing_dataset])
    X= pd.concat([complete_dataset.iloc[:, 0:3], complete_dataset.iloc[:, 4:]], axis=1 ).to_numpy()
    y= complete_dataset.iloc[:, 3].to_numpy()

    #for classification
    training_dataset = pd.read_csv('../data/classification/4_classes/training_data.csv')
    testing_dataset = pd.read_csv('../data/classification/4_classes/testing_data.csv')
    complete_dataset = pd.concat([training_dataset, testing_dataset])
    y_classification= complete_dataset.iloc[:, 3].to_numpy()

    return X, y, y_classification

def regression_predict(data, X, y):
    """
    Predicts using Linear Regression and KNN-regression, returns a dictionary
    {
    'linear regression': value,
    'KNN regression':value
    }
    """
    data = np.array(list(data.values()))
    #linear regression
    with open('../cache/linear_regression.pkl', 'rb') as f:
        parameters = np.array(pickle.load(f)["parameters"])

    linear_regression_prediction = np.dot(parameters[1:], data) + parameters[0]

    #knn regression
    with open ('../cache/knn_regression.pkl', 'rb') as f:
        k = pickle.load(f)['k']

    knn_prediction= KNN_prediction(k, X, y, data)[0]

    results = {
        'linear regression': linear_regression_prediction,
        'KNN regression':knn_prediction
    }
    return results

def classification_predict(data, X, y):
    """
    Predicts the probability of every class using Logistic Regression, KNN Classification and Generative Models returns 4 dictionaries 
    {'cls': probability}
    (for logistic regression there will be only 2 classes)
    """
    data_series = pd.Series(data)
    data = np.array(list(data.values()))
    #logistic regression
    with open('../cache/logistic_regression.pkl', 'rb') as f:
        parameters = np.array(pickle.load(f)["parameters"])

    data_bias = np.insert(data, 0, 1)
    probability_expensive = float(logistic_function(data_bias, parameters))

    log_regression = {
        'cheap' : 1.0-probability_expensive,
        'expensive': probability_expensive
    }

    #KNN classification
    with open('../cache/knn_classification.pkl', 'rb') as f:
        k = pickle.load(f)['k']
    
    knn = K_nearest_neighbors(k, X, y, data)

    #generative models
    means = np.load('../cache/means.npy')
    covariance_matrices = np.load('../cache/covariance_matrices.npy')
    pooled_covariance = np.load('../cache/pooled_covariance.npy')
    prior_proba =  np.load('../cache/prior_probabilities.npy')
    lda = discriminant_analysis_one_point(data, means,pooled_covariance, prior_proba, True)
    qda = discriminant_analysis_one_point(data, means,covariance_matrices, prior_proba, False)

    with open('../cache/mean_std_estimates.pkl', 'rb') as f:
        mean_std_estimates = pickle.load(f)
    naive_bayes = naive_bayes_probabilities(data_series, mean_std_estimates, prior_proba)

    return log_regression, knn, lda, qda, naive_bayes


def print_info(regression, log_regression, knn, lda, qda, naive_bayes):
    """
    Disclaimer: this function was (the only one) generated by AI tools, as it doesn't contribute to the any of the ML algorithms, and is a frontend interface only.
    """
    st.divider()

    st.header("📊 Prediction Results")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("📈 Regression")

        with st.container(border=True):
            st.markdown("### Linear Regression")
            st.metric(
                label="Predicted Price",
                value=f"${regression['linear regression']:,.0f}"
            )

        with st.container(border=True):
            st.markdown("### KNN Regression")
            st.metric(
                label="Predicted Price",
                value=f"${regression['KNN regression']:,.0f}"
            )


    with col2:
        st.subheader(" Binary Classification")

        with st.container(border=True):
            st.markdown("### Logistic Regression")

            # Cast values explicitly to float
            st.progress(float(log_regression["cheap"]))
            st.write(f"Cheap: **{log_regression['cheap']:.2%}**")

            st.progress(float(log_regression["expensive"]))
            # FIX: Added missing closing '**' at the end of the string below
            st.write(f"Expensive: **{log_regression['expensive']:.2%}**")

    with col3:
        st.subheader(" Multi-class Classification")

        for name, result in [
            ("KNN", knn),
            ("LDA", lda),
            ("QDA", qda),
            ("Naive Bayes", naive_bayes),
        ]:

            with st.expander(name, expanded=True):

                # Cast NumPy values explicitly to float
                st.progress(float(result["cheap"]))
                st.write(f"Cheap: **{result['cheap']:.2%}**")

                st.progress(float(result["moderate"]))
                st.write(f"Moderate: **{result['moderate']:.2%}**")

                st.progress(float(result["expensive"]))
                st.write(f"Expensive: **{result['expensive']:.2%}**")

                st.progress(float(result["very expensive"]))
                st.write(f"Very Expensive: **{result['very expensive']:.2%}**")