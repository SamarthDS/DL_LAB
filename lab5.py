# Install necessary packages (uncomment if needed in a new environment)
# !pip install numpy tensorflow matplotlib

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Dummy time series data
data = np.sin(np.linspace(0, 100, 1000))
X = np.array([data[i-10:i] for i in range(10, len(data))])[..., None]
y = data[10:]

# LSTM model
lstm = tf.keras.Sequential([
    tf.keras.layers.LSTM(32, input_shape=(10, 1)),
    tf.keras.layers.Dense(1)
])
lstm.compile(optimizer='adam', loss='mse')
lstm_history = lstm.fit(X, y, epochs=3, verbose=0)
lstm_loss = lstm_history.history['loss'][-1]
print(f"LSTM Loss: {lstm_loss}")

# GRU model
gru = tf.keras.Sequential([
    tf.keras.layers.GRU(32, input_shape=(10, 1)),
    tf.keras.layers.Dense(1)
])
gru.compile(optimizer='adam', loss='mse')
gru_history = gru.fit(X, y, epochs=3, verbose=0)
gru_loss = gru_history.history['loss'][-1]
print(f"GRU Loss: {gru_loss}")

# Optional: Plot loss over epochs for both models
plt.plot(lstm_history.history['loss'], label='LSTM')
plt.plot(gru_history.history['loss'], label='GRU')
plt.title('Model Loss Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()