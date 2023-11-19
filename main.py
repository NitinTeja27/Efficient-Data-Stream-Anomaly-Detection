import numpy as np
from data_generator import DataGenerator
from z_score import Z_Score
import matplotlib.pyplot as plt
from exponential_smoothing import ExponentialSmoothing

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

    plt.show()

if __name__ == "__main__":
    main()