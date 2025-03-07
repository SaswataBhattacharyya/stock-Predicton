import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM, Dropout
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

if not tf.config.list_physical_devices('GPU'):
    st.warning("⚠️ No GPU found. Running on CPU.")
else:
    st.success("✅ Running on GPU.")

# Load dataset function
def load_data(uploaded_file):
    df = pd.read_excel(uploaded_file) if uploaded_file.name.endswith('.xlsx') else pd.read_csv(uploaded_file)
    df['Date'] = pd.to_datetime(df['Date'])
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month

    start_date = df['Date'].min()
    end_date = df['Date'].max()
    full_date_range = pd.date_range(start=start_date, end=end_date)

    df = df.set_index('Date').reindex(full_date_range)
    df['Year'] = df.index.year  # Ensure Year column is updated after reindexing
    df.ffill(inplace=True)  # Forward fill first
    df.bfill(inplace=True)  # Then backward fill
    df.reset_index(inplace=True)
    df.rename(columns={'index': 'Date'}, inplace=True)

    df.dropna(subset=['Date'], inplace=True)
    return df

# Preprocess data
def preprocess_data(df):
    scaler = MinMaxScaler()
    df['Normalized_Close'] = scaler.fit_transform(df[['Close']])
    df['Smoothed_Close'] = df['Normalized_Close'].rolling(window=30, min_periods=1).mean()
    
    # 🚨 Store min and max values for later inverse transformation
    X_min = df['Close'].min()
    X_max = df['Close'].max() 
    
    values = df['Smoothed_Close'].dropna().values.reshape(-1, 1)
    scaled_values = scaler.fit_transform(values)

    X, y = [], []
    seq_length = 30
    for i in range(len(scaled_values) - seq_length):
        X.append(scaled_values[i:i + seq_length])
        y.append(scaled_values[i + seq_length])

    X, y = np.array(X), np.array(y)

    return X, y, scaler, X_min, X_max 

# Build LSTM model
def build_lstm_model(input_shape):
    model = Sequential([
        LSTM(units=100, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(units=50, return_sequences=False),
        Dropout(0.2),
        Dense(units=25, activation='relu'),
        Dense(units=1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Predict future values
def predict_future(model, df, scaler, days_ahead, X_min, X_max): 
    last_sequence = df['Smoothed_Close'].dropna().iloc[-30:].values.reshape(1, -1, 1)
    predictions = []

    for _ in range(days_ahead):
        next_pred = model.predict(last_sequence).reshape(1, 1, 1)
        predictions.append(next_pred[0, 0])
        last_sequence = np.append(last_sequence[:, 1:, :], next_pred, axis=1)

    future_dates = pd.date_range(df['Date'].iloc[-1] + pd.Timedelta(days=1), periods=days_ahead)
    
    predicted_prices = np.array(predictions).reshape(-1, 1)
    
    # manual inverse transformation
    predicted_prices = X_min + (predicted_prices * (X_max - X_min))

    return future_dates, predicted_prices

# Streamlit App
st.title("Stock Price Prediction using LSTM")

uploaded_file = st.file_uploader("Upload an Excel or CSV file", type=["xlsx", "csv"])

if uploaded_file:
    df = load_data(uploaded_file)
    X, y, scaler, X_min, X_max = preprocess_data(df)

    # Train the model
    model = build_lstm_model((X.shape[1], X.shape[2]))
    model.fit(X, y, epochs=25, batch_size=32, verbose=0)

    start_year, end_year = df['Year'].min(), df['Year'].max()
    
    actual_range = st.slider("Select Actual Data Range (Years)", start_year, end_year, (start_year, end_year))
    days_ahead = st.slider("Predict Future Days", 1, 60, 30)
    
    df_filtered = df[(df['Year'] >= actual_range[0]) & (df['Year'] <= actual_range[1])]
    
    future_dates, future_predictions = predict_future(model, df, scaler, days_ahead, X_min, X_max)

    # Plot results
    st.write("### Actual vs Predicted Prices")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df_filtered['Date'], df_filtered['Close'], label='Actual')
    ax.plot(future_dates, future_predictions, label='Predicted', linestyle='dashed', color='red')
    ax.set_xlabel("Date")
    ax.set_ylabel("Stock Price")
    ax.legend()
    st.pyplot(fig)

    # Display table of predictions
    st.write("### Predicted Values")
    prediction_df = pd.DataFrame({
    "Date": future_dates, 
    "Predicted Price": np.array(future_predictions).reshape(-1)  # Force to 1D
})
    st.dataframe(prediction_df)
