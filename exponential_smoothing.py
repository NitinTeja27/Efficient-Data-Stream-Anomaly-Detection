import matplotlib.pyplot as plt

class ExponentialSmoothing:
    def __init__(self, alpha=0.3, threshold=2.0):
        self.alpha = alpha 
        self.threshold = threshold 
        self.smoothed_value = None
    
    def detect_anomaly(self, new_value):
        if self.smoothed_value is None:
            self.smoothed_value = new_value
            return False, self.smoothed_value
        
        self.smoothed_value = self.alpha * new_value + (1 - self.alpha) * self.smoothed_value
        deviation = abs(new_value - self.smoothed_value)
        is_anomaly = deviation > self.threshold
        return is_anomaly, self.smoothed_value
    
    def plot_data(self, data_stream, anomaly_indices):
        fig = plt.figure()
        plt.plot(data_stream, label='Data Stream')
        plt.scatter(anomaly_indices, data_stream[anomaly_indices], color='red', label='Anomalies')
        plt.title('Anomaly Detection using Exponential Smoothing')
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.legend()
        return fig

    def simulate_Exponential_Smoothing(self,  data_stream):
        anomaly_indices = []
        for i, data_point in enumerate(data_stream):
            is_anomaly, smoothed_value = self.detect_anomaly(data_point)
            if is_anomaly:
                anomaly_indices.append(i)

        fig = self.plot_data(data_stream=data_stream, anomaly_indices=anomaly_indices)
        return fig