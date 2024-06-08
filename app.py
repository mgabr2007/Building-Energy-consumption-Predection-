import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np

# Function to load data
def load_data(uploaded_file):
    return pd.read_excel(uploaded_file)

# Function to train a simple linear regression model
def train_model(df):
    df['Year'] = df['year']
    df['Yearly_Consumption'] = df['yearly_consumption']

    X = df[['Year']]
    y = df['Yearly_Consumption']
    model = LinearRegression()
    model.fit(X, y)
    return model

# Function to predict future consumption
def predict_future(model, start_year, periods):
    future_years = np.arange(start_year, start_year + periods).reshape(-1, 1)
    future_consumption = model.predict(future_years)
    future_df = pd.DataFrame({
        'Year': future_years.flatten(),
        'Predicted Consumption': future_consumption
    })
    return future_df

# Streamlit UI
st.title("Building Consumption Data Analysis and Prediction")

uploaded_file = st.file_uploader("Upload your Excel file", type=["xlsx"])

if uploaded_file is not None:
    df = load_data(uploaded_file)
    st.write("Uploaded Data:")
    st.dataframe(df)

    model = train_model(df)
    if model is not None:
        st.write("Training complete. Model coefficients:")
        st.write(f"Intercept: {model.intercept_}")
        st.write(f"Coefficients: {model.coef_}")

        start_year = df['year'].max() + 1
        periods = st.slider("Select number of years to predict into the future", 1, 24, 12)
        future_df = predict_future(model, start_year, periods)

        st.write("Future Predictions:")
        st.dataframe(future_df)

        fig, ax = plt.subplots()
        ax.plot(df['year'], df['yearly_consumption'], label='Historical Consumption')
        ax.plot(future_df['Year'], future_df['Predicted Consumption'], label='Predicted Consumption', linestyle='--')
        ax.set_xlabel("Year")
        ax.set_ylabel("Consumption")
        ax.legend()
        st.pyplot(fig)

st.write("Please upload an Excel file to get started.")
