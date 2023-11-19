import numpy as np
import matplotlib.pyplot as plt

from data_generator import DataGenerator
from z_score import Z_Score
from exponential_smoothing import ExponentialSmoothing
from isolation_forest import _IsolationForest_
from autoencoder import Autoencoder

def main():
    num_points = 1000
    offset = 5
    data_gen = DataGenerator()
    data_stream = data_gen.simulate_data_stream(num_points=num_points,offset=offset)
    window_size = 50
    threshold = 3

    z_sc = Z_Score()
    fig_z_sc = z_sc.simulate_Z_score_detection(data_stream, window_size, threshold)
    
    exponen_sc = ExponentialSmoothing()
    fig_exponen_sc = exponen_sc.simulate_Exponential_Smoothing(data_stream=data_stream)

    isolation_forest = _IsolationForest_()
    fig_isolation_forest = isolation_forest.simulate_Isolation_Forest(data_stream=data_stream)

    input_dim = 1  
    autoencoder = Autoencoder(input_dim=input_dim, encoding_dim=8)
    fig = autoencoder.simulate_autoencoder(data_stream=data_stream)

    plt.show()

if __name__ == "__main__":
    main()