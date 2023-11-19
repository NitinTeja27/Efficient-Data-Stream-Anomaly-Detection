
from sklearn.ensemble import IsolationForest
import numpy as np
import matplotlib.pyplot as plt
from data_generator import DataGenerator

class _IsolationForest_:
    def __init__(self, n_estimators=100, contamination=0.1):
        self.detector = IsolationForest(n_estimators=n_estimators, contamination=contamination)
    
    def train(self, X_train):
        self.detector.fit(X_train)
    
    def detect_anomaly(self, new_value):
        is_anomaly = self.detector.predict([[new_value]])
        return is_anomaly == -1
    
    def plot_data(self, data_stream, anomaly_indices):
        fig = plt.figure()
        plt.plot(data_stream, label='Data Stream')
        plt.scatter(anomaly_indices, data_stream[anomaly_indices], color='red', label='Anomalies')
        plt.title('Anomaly Detection using Isolation Forest')
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.legend()
        return fig
    
    def simulate_Isolation_Forest(self,  data_stream):
        anomaly_indices = []
        
        train_data = np.random.normal(0, 1, (1000, 1))
        self.train(train_data)
        
        for i, value in enumerate(data_stream):
            is_anomaly = self.detect_anomaly(value)
            if is_anomaly:
                anomaly_indices.append(i)
        fig = self.plot_data(data_stream=data_stream, anomaly_indices=anomaly_indices)
        return fig