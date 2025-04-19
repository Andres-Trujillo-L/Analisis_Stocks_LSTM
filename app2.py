import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import numpy as np
import yfinance as yf
import ta
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense
from pandas.tseries.offsets import BDay
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os

st.set_page_config(layout="wide")
st.title("\U0001F4C8 Análisis de Acciones - Datos de Yahoo Finance")
st.markdown("Esta app permite analizar acciones con indicadores técnicos y generar señales de compra/venta actualizadas.")

# Sidebar
st.sidebar.subheader("\U0001F50E Buscar acción (por símbolo de Yahoo Finance)")
query = st.sidebar.text_input("Buscar acción:", value="TSLA")

start_date = st.sidebar.date_input("Desde", datetime.date(2021, 1, 1))
end_date = st.sidebar.date_input("Hasta", datetime.date.today())

@st.cache_data(ttl=3600)
def load_data_yf(symbol, start_date, end_date):
    try:
        df = yf.download(symbol, start=start_date, end=end_date)
        df.reset_index(inplace=True)
        return df
    except Exception as e:
        st.error(f"Error al obtener datos: {e}")
        return pd.DataFrame()

df = load_data_yf(query, start_date, end_date)

if df.empty:
    st.warning("No se pudieron cargar los datos. Revisa el símbolo o las fechas.")
    st.stop()

# Indicadores técnicos
df['SMA_20'] = df['Close'].rolling(20).mean()
df['SMA_50'] = df['Close'].rolling(50).mean()
df['RSI'] = ta.momentum.RSIIndicator(close=df['Close'].squeeze()).rsi()
macd = ta.trend.MACD(close=df['Close'].squeeze())
df['MACD'] = macd.macd()
df['MACD_Signal'] = macd.macd_signal()
df['MACD_Diff'] = macd.macd_diff()

# Gráfico de evolución
st.subheader(f"\U0001F4CA Evolución del Precio de {query}")
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(df['Date'], df['Close'], label='Precio Cierre', color='blue')
ax.plot(df['Date'], df['SMA_20'], label='SMA 20', linestyle='--')
ax.plot(df['Date'], df['SMA_50'], label='SMA 50', linestyle='--')
ax.set_title(f'{query} - Precio con Medias Móviles')
ax.legend()
st.pyplot(fig)

# Señales
st.subheader("\U0001F4CC Señales Técnicas")
if df['RSI'].iloc[-1] < 30:
    st.info("\U0001F53C RSI indica posible COMPRA (zona de sobreventa)")
elif df['RSI'].iloc[-1] > 70:
    st.info("\U0001F53D RSI indica posible VENTA (zona de sobrecompra)")
else:
    st.info("RSI en zona neutral")

if df['SMA_20'].iloc[-1] > df['SMA_50'].iloc[-1]:
    st.success("\U0001F4C8 Cruce de medias indica tendencia ALCISTA")
else:
    st.warning("\U0001F4C9 Cruce de medias indica tendencia BAJISTA")

st.subheader("\U0001F4CB Últimos datos")
st.dataframe(df.tail(10))

# Preparación para LSTM multivariado
st.subheader("\U0001F52E Predicción con LSTM para los próximos 15 días hábiles (Multivariado)")
features = ['Close', 'Volume', 'RSI', 'MACD', 'MACD_Signal']
df[features] = df[features].fillna(method='bfill')
scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(df[features])

sequence_length = 50
X_lstm, y_lstm = [], []
for i in range(sequence_length, len(scaled_features)):
    X_lstm.append(scaled_features[i-sequence_length:i])
    y_lstm.append(scaled_features[i, 0])  # 0 es 'Close'
X_lstm, y_lstm = np.array(X_lstm), np.array(y_lstm)

split_idx = int(len(X_lstm) * 0.8)
X_train, X_test = X_lstm[:split_idx], X_lstm[split_idx:]
y_train, y_test = y_lstm[:split_idx], y_lstm[split_idx:]

@st.cache_resource
def train_or_load_model():
    model_path = f"lstm_model_multivariate_{query}.h5"
    if os.path.exists(model_path):
        return load_model(model_path)
    else:
        model = Sequential()
        model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
        model.add(LSTM(units=50))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)
        model.save(model_path)
        return model

model = train_or_load_model()
predicted_test = model.predict(X_test)
actual_test_close = scaler.inverse_transform(np.concatenate((predicted_test, X_test[:, -1, 1:]), axis=1))[:, 0]
y_test_reshaped = y_test.reshape(-1, 1)
y_test_rescaled = scaler.inverse_transform(np.concatenate((y_test_reshaped, X_test[:, -1, 1:]), axis=1))[:, 0]

# Métricas
rmse = np.sqrt(mean_squared_error(y_test_rescaled, actual_test_close))
mae = mean_absolute_error(y_test_rescaled, actual_test_close)
r2 = r2_score(y_test_rescaled, actual_test_close)

st.metric(label="\U0001F4C9 RMSE (Test Set)", value=f"{rmse:.2f}")
st.metric(label="\U0001F4CA MAE (Test Set)", value=f"{mae:.2f}")
st.metric(label="\U0001F50E R² (Test Set)", value=f"{r2:.2f}")

# Predicción futura
input_seq = scaled_features[-sequence_length:]
predictions = []
for _ in range(15):
    X_pred = input_seq[-sequence_length:]
    X_pred = X_pred.reshape(1, sequence_length, X_lstm.shape[2])
    pred = model.predict(X_pred, verbose=0)
    pred_full = np.append(pred, X_pred[0, -1, 1:])
    predictions.append(pred_full)
    input_seq = np.vstack([input_seq, pred_full])

predicted_array = np.array(predictions)
predicted_close = scaler.inverse_transform(predicted_array)[:, 0]
future_dates = [df['Date'].max() + BDay(i) for i in range(1, 16)]
df_future = pd.DataFrame({
    'Date': future_dates,
    'Predicted_Close': predicted_close
})

fig2, ax2 = plt.subplots(figsize=(12, 5))
ax2.plot(df['Date'], df['Close'], label='Histórico', color='blue')
ax2.plot(df_future['Date'], df_future['Predicted_Close'], label='LSTM Predicción', color='green', linestyle='--')
ax2.set_title(f'Predicción de {query} para los próximos 15 días hábiles (LSTM Multivariado)')
ax2.legend()
st.pyplot(fig2)

# Señal basada en la predicción
st.subheader("\U0001F4CC Señales basadas en predicción")
if df_future['Predicted_Close'].values[-1] > df['Close'].values[-1]:
    st.success("\U0001F4C8 Se predice una subida en los próximos 15 días. Posible señal de COMPRA.")
else:
    st.error("\U0001F4C9 Se predice una caída en los próximos 15 días. Posible señal de VENTA o cautela.")

st.dataframe(df_future)
