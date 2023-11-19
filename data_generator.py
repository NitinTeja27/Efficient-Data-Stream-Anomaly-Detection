import numpy as np

class DataGenerator:
    def __init__(self) -> None:
        pass
    
    def simulate_data_stream(self, num_points=1000, offset=5):
        np.random.seed(42)
        data_stream = np.random.normal(0, 1, num_points)
        anomaly_indices = np.random.choice(num_points, size=10, replace=False)
        data_stream[anomaly_indices] += offset 
        return data_stream