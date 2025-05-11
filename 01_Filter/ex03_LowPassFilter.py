import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class LowPassFilter:
    def __init__(self, y_initial_measure, alpha=0.1):
        self.y_estimate = y_initial_measure  # 초기 추정값
        self.alpha = alpha                   # 필터 계수 α (과거에 줄 비중)

    def estimate(self, y_measure):
        self.y_estimate = self.alpha * self.y_estimate + (1 - self.alpha) * y_measure

if __name__ == "__main__":
    #signal = pd.read_csv("01_filter/Data/example_Filter_1.csv")
    #signal = pd.read_csv("01_filter/Data/example_Filter_2.csv")      
    signal = pd.read_csv("01_filter/Data/example_Filter_3.csv")

    signal["y_estimate"] = 0.0  # 추정값 컬럼 생성

    y_filter = LowPassFilter(signal.y_measure[0], alpha=0.9)
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
