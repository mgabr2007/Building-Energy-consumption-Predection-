import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np

# Function to load data
def load_data(uploaded_file):
    return pd.read_excel(uploaded_file)

# Function to train a simple linear regression model
def train_model(df):
    date_col = None
    for col in df.columns:
        if 'date' in col.lower():
            date_col = col
            break
    if date_col is None:
        st.error("No column containing 'date' found in the uploaded file.")
        return None

    df[date_col] = pd.to_datetime(df[date_col])
    df.set_index(date_col, inplace=True)
    df['Month'] = df.index.month
    df['Year'] = df.index.year

    X = df[['Month', 'Year']]
    y = df['Consumption']
    model = LinearRegression()
    model.fit(X, y)
    return model

# Function to predict future consumption
def predict_future(model, periods):
    last_date = datetime.now()
    future_dates = [last_date + pd.DateOffset(months=i) for i in range(1, periods + 1)]
    future_df = pd.DataFrame({
        'Month': [date.month for date in future_dates],
        'Year': [date.year for date in future_dates]
    })
    future_df['Predicted Consumption'] = model.predict(future_df)
    future_df['Date'] = future_dates
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

        periods = st.slider("Select number of months to predict into the future", 1, 24, 12)
        future_df = predict_future(model, periods)

        st.write("Future Predictions:")
        st.dataframe(future_df[['Date', 'Predicted Consumption']])

        fig, ax = plt.subplots()
        ax.plot(df.index, df['Consumption'], label='Historical Consumption')
        ax.plot(future_df['Date'], future_df['Predicted Consumption'], label='Predicted Consumption', linestyle='--')
        ax.set_xlabel("Date")
        ax.set_ylabel("Consumption")
        ax.legend()
        st.pyplot(fig)

st.write("Please upload an Excel file to get started.")
