import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class AverageFilter:
    def __init__(self, y_initial_measure):
        self.y_estimate = y_initial_measure  # 초기 추정값 설정
        self.k = 1  # 측정 횟수 초기화

    def estimate(self, y_measure):
        self.k += 1  # 새로운 측정이 들어올 때마다 k 증가
        self.y_estimate = ((self.k - 1) / self.k) * self.y_estimate + (1 / self.k) * y_measure

if __name__ == "__main__":
    signal = pd.read_csv("01_filter/Data/example_Filter_1.csv")
    signal["y_estimate"] = 0.0  # 추정값 컬럼 초기화

    y_filter = AverageFilter(signal.y_measure[0])
    signal.loc[0, "y_estimate"] = signal.y_measure[0]

    for i in range(1, len(signal)):
        y_filter.estimate(signal.y_measure[i])
        signal.loc[i, "y_estimate"] = y_filter.y_estimate

    plt.figure()
    plt.plot(signal.time, signal.y_measure, 'k.', label="Measure")
    plt.plot(signal.time, signal.y_estimate, 'r-', label="Estimate")
    plt.xlabel('time (s)')
    plt.ylabel('signal')
    plt.legend(loc="best")
    plt.axis("equal")
    plt.grid(True)
    plt.show()
