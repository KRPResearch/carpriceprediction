import streamlit as st
import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

# Title
st.title("🚗 Used Car Price Prediction App (Multiple Models)")

# Load dataset
@st.cache_data
def load_data():
    data = pd.read_csv("usedcarsales.csv")
    data = pd.get_dummies(data, columns=['FuelType', 'AutoType', 'MetColorType'])
    return data

carsales = load_data()

# Prepare data
y = carsales['Price']
X = carsales.drop(columns=['Price'])

# Train models
@st.cache_resource
def train_models(X, y):
    models = {}

    models["Linear Regression"] = LinearRegression().fit(X, y)
    models["Decision Tree"] = DecisionTreeRegressor().fit(X, y)
    models["Random Forest"] = RandomForestRegressor().fit(X, y)
    models["Gradient Boosting"] = GradientBoostingRegressor().fit(X, y)

    return models

models = train_models(X, y)

# Model selection
st.subheader("Select Model")
model_name = st.selectbox(
    "Choose a model",
    list(models.keys())
)

model = models[model_name]

# Input section
st.subheader("Enter Car Details")

input_data = {}
for col in X.columns:
    input_data[col] = st.number_input(f"{col}", value=0.0)

# Convert input to dataframe
input_df = pd.DataFrame([input_data])

# Align columns
input_df = input_df.reindex(columns=X.columns, fill_value=0)

# Prediction
if st.button("Predict Price"):
    prediction = model.predict(input_df)
    st.success(f"💰 Predicted Price ({model_name}): {prediction[0]:.2f}")
    st.success(f"Predicted Car Price: {prediction[0]:.2f}")