import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Load model and scaler
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

# Load historical stock data
df = pd.read_csv("stock_data.csv")
df['Date'] = pd.to_datetime(df['Date'])
df['Price'] = pd.to_numeric(df['Price'], errors='coerce')
df['Open'] = pd.to_numeric(df['Open'], errors='coerce')
df['High'] = pd.to_numeric(df['High'], errors='coerce')
df['Low'] = pd.to_numeric(df['Low'], errors='coerce')
df = df.dropna()

# Title
st.title(" Stock Movement Predictor (Up/Down)")
st.markdown("Predict whether the stock will go up or down with % confidence")

# Plot with regression line
st.subheader("Price Over Time with Regression Line")
fig, ax = plt.subplots(figsize=(10, 4))
sns.regplot(data=df, x=df.index, y="Price", scatter_kws={"s": 10}, ax=ax, line_kws={"color": "red"})
ax.set_xlabel("Time Index")
ax.set_ylabel("Price")
st.pyplot(fig)

# Input fields
st.subheader("Enter Stock Data for Today")

open_val = st.number_input("Open Price", value=float(df.iloc[-1]['Open']))
high_val = st.number_input("High Price", value=float(df.iloc[-1]['High']))
low_val = st.number_input("Low Price", value=float(df.iloc[-1]['Low']))

# Prediction logic
if st.button(" Predict Movement"):
    # Use last 10 rows for moving average calculation
    recent_df = df[['Open', 'High', 'Low', 'Price']].copy().iloc[-10:].copy()

    # Append the new day's data
    today_price = (high_val + low_val) / 2  # Approximate current day's price
    new_row = {
        'Open': open_val,
        'High': high_val,
        'Low': low_val,
        'Price': today_price
    }

    recent_df = pd.concat([recent_df, pd.DataFrame([new_row])], ignore_index=True)

    # Recalculate indicators
    recent_df['MA5'] = recent_df['Price'].rolling(window=5).mean()
    recent_df['MA10'] = recent_df['Price'].rolling(window=10).mean()
    recent_df['Price_Change'] = recent_df['Price'].pct_change()

    latest = recent_df.iloc[-1][['Open', 'High', 'Low', 'MA5', 'MA10', 'Price_Change']]

    # Check if any features are missing
    if latest.isnull().any():
        st.error(" Not enough historical data to compute indicators. Please use more data.")
    else:
        # Predict
        input_scaled = scaler.transform([latest.values])
        pred = model.predict(input_scaled)[0]
        prob = model.predict_proba(input_scaled)[0]

        # Show results
        label = " Stock will go UP" if pred == 1 else " Stock will go DOWN"
        st.markdown(f"### {label}")
        st.markdown(f"**Probability Up:** `{prob[1]*100:.2f}%`")
        st.markdown(f"**Probability Down:** `{prob[0]*100:.2f}%`")

        # Plot confidence
        st.subheader("Prediction Confidence")
        conf_df = pd.DataFrame({
            'Movement': ['Down', 'Up'],
            'Probability': [prob[0], prob[1]]
        })

        fig2, ax2 = plt.subplots()
        sns.barplot(data=conf_df, x='Movement', y='Probability', palette=['red', 'green'], ax=ax2)
        ax2.set_ylim(0, 1)
        ax2.set_ylabel('Probability')
        ax2.set_title('Model Confidence')
        st.pyplot(fig2)
