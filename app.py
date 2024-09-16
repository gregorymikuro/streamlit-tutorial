import streamlit as st
import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import pickle

# # load dataset
# iris = load_iris()
#
# X, y = iris.data, iris.target
#
# # train the model
# model = RandomForestClassifier()
#
# model.fit(X, y)
#
#
# # save the model
# with open('iris_model.pkl', 'wb') as f:
#     pickle.dump(model, f)


# load the model from the pickle file
with open('iris_model.pkl', 'rb') as f:
    model = pickle.load(f)

iris = load_iris()

# create a prediction function
def predict_iris(features):
    return model.predict([features])


st.title("Iris Dataset Flower Prediction")


# input features
sepal_length = st.slider('Sepal Length', 4.0, 8.0, 5.0)
sepal_width = st.slider('Sepal Width', 2.0, 4.4, 3.0)
petal_length = st.slider('Petal Length', 1.0, 7.0, 3.5)
petal_width = st.slider('Petal Width', 0.1, 2.5, 1.2)

# Prediction button
if st.button('Predict'):
    features = [sepal_length, sepal_width, petal_length, petal_width]
    prediction = predict_iris(features)
    st.write(f"The prediction class is {iris.target_names[prediction][0]}")