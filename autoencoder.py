import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt

from data_generator import DataGenerator

class Autoencoder:
    def __init__(self, input_dim, encoding_dim=8):
        input_layer = Input(shape=(input_dim,))
        encoded = Dense(encoding_dim, activation='relu')(input_layer)
        decoded = Dense(input_dim, activation='sigmoid')(encoded)
        
        self.autoencoder = Model(input_layer, decoded)
        self.autoencoder.compile(optimizer='adam', loss='mse')
    
    def train(self, X_train, epochs=10, batch_size=32):
        self.autoencoder.fit(X_train, X_train, epochs=epochs, batch_size=batch_size, shuffle=True)
    
    def detect_anomaly(self, new_value):
        reconstructed_value = self.autoencoder.predict(np.array([new_value]))
        loss = np.mean(np.square(new_value - reconstructed_value))
        return loss > 0.1  
    
    def plot_data(self, data_stream, anomaly_indices):
        fig = plt.figure()
        plt.plot(data_stream, label='Data Stream')
        plt.scatter(anomaly_indices, data_stream[anomaly_indices], color='red', label='Anomalies')
        plt.title('Anomaly Detection using Autoencoder')
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.legend()
        return fig
    
    def simulate_autoencoder(self, data_stream):
        anomaly_indices = []
        train_data = np.random.normal(0, 1, (1000, 1))
        input_dim = 1  
        self.train(train_data, epochs=10, batch_size=32)

        for i, value in enumerate(data_stream): 
            new_value = value
            is_anomaly = self.detect_anomaly(new_value)
            if is_anomaly:
                anomaly_indices.append(i)

        fig = self.plot_data(data_stream=data_stream, anomaly_indices=anomaly_indices)
        return fig