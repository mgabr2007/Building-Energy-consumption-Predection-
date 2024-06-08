import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np

# Function to load data
def load_data(uploaded_file):
    return pd.read_excel(uploaded_file)

# Function to train a simple linear regression model
def train_model(df, selected_month=None):
    df['Year'] = df['year']
    if selected_month is not None:
        df['Monthly_Consumption'] = df[f'value_{selected_month}']
        X = df[['Year']]
        y = df['Monthly_Consumption']
    else:
        df['Yearly_Consumption'] = df['yearly_consumption']
        X = df[['Year']]
        y = df['Yearly_Consumption']

    model = LinearRegression()
    model.fit(X, y)
    return model

# Function to predict future consumption
def predict_future(model, start_year, periods, selected_month=None):
    future_years = np.arange(start_year, start_year + periods).reshape(-1, 1)
    future_consumption = model.predict(future_years)
    future_df = pd.DataFrame({
        'Year': future_years.flatten(),
        'Predicted Consumption': future_consumption
    })
    return future_df

# Streamlit UI
st.title("Building Consumption Data Analysis and Prediction")

st.write("""
### Process Overview

1. **Upload Data:** Upload your Excel file containing the building's energy consumption data.
2. **Data Analysis:** The app reads the data and displays the uploaded information.
3. **Model Training:** A linear regression model is trained using the historical consumption data (either yearly or monthly).
4. **Future Prediction:** Based on the trained model, the app predicts future energy consumption for a specified number of years.

### Expected Outcome

- **Historical Data Display:** View the uploaded historical consumption data in a table format.
- **Model Details:** See the coefficients of the trained linear regression model.
- **Future Predictions:** Get predictions for future energy consumption, displayed in both a table and a graphical format.
""")

uploaded_file = st.file_uploader("Upload your Excel file", type=["xlsx"])

if uploaded_file is not None:
    df = load_data(uploaded_file)
    st.write("Uploaded Data:")
    st.dataframe(df)

    analysis_type = st.radio("Select analysis type:", ("Total Yearly Consumption", "Monthly Consumption"))

    if analysis_type == "Monthly Consumption":
        selected_month = st.selectbox("Select month to analyze (1-12):", list(range(1, 13)))
        model = train_model(df, selected_month)
    else:
        selected_month = None
        model = train_model(df)

    if model is not None:
        st.write("Training complete. Model coefficients:")
        st.write(f"Intercept: {model.intercept_}")
        st.write(f"Coefficients: {model.coef_}")

        st.write("""
        ### Understanding Model Coefficients

        The linear regression model provides us with two key parameters:
        1. **Intercept:** This is the baseline value of the energy consumption when the year is zero. It essentially represents the starting point of our prediction line.
        2. **Coefficients:** These represent the change in energy consumption for each additional year. In other words, the coefficient indicates how much the energy consumption is expected to increase or decrease per year.

        For example, if the coefficient is 5000, it means that each year, the energy consumption is expected to increase by 5000 units.
        """)

        start_year = df['year'].max() + 1
        periods = st.slider("Select number of years to predict into the future", 1, 24, 12)
        future_df = predict_future(model, start_year, periods, selected_month)

        st.write("Future Predictions:")
        st.dataframe(future_df)

        fig, ax = plt.subplots()
        if selected_month is not None:
            ax.plot(df['year'], df[f'value_{selected_month}'], label='Historical Monthly Consumption')
        else:
            ax.plot(df['year'], df['yearly_consumption'], label='Historical Yearly Consumption')
        ax.plot(future_df['Year'], future_df['Predicted Consumption'], label='Predicted Consumption', linestyle='--')
        ax.set_xlabel("Year")
        ax.set_ylabel("Consumption")
        ax.legend()
        st.pyplot(fig)

st.write("Please upload an Excel file to get started.")
