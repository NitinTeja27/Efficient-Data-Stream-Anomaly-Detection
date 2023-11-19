import numpy as np
from scipy.stats import zscore
import matplotlib.pyplot as plt
from collections import deque

class Z_Score:
    def __init__(self) -> None:
        pass

    def detect_anomalies(self, data, threshold=3):
        z_scores = zscore(data)
        anomalies = np.where(np.abs(z_scores) > threshold)[0]
        return anomalies

    def plot_data(self, data_stream, anomaly_indices):
        fig = plt.figure()
        plt.plot(data_stream, label='Data Stream')
        plt.scatter(anomaly_indices, data_stream[anomaly_indices], color='red', label='Anomalies')
        plt.title('Anomaly Detection using Moving Z-Score')
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.legend()
        return fig

    def simulate_Z_score_detection(self,  data_stream, window_size, threshold):
        data_buffer = deque(maxlen=window_size)
        anomaly_indices = []
        for i, data_point in enumerate(data_stream):
            data_buffer.append(data_point)
            if len(data_buffer) == window_size:
                anomalies = self.detect_anomalies(np.array(data_buffer), threshold)
                anomalies += i - window_size + 1
                anomaly_indices.extend(anomalies)

        fig = self.plot_data(data_stream=data_stream, anomaly_indices=anomaly_indices)
        return fig