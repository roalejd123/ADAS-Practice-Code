from ex01_AverageFilter import AverageFilter
from ex02_MovingAverageFilter import MovingAverageFilter
from ex03_LowPassFilter import LowPassFilter
from ex04_KalmanFilter import KalmanFilter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == "__main__":
    t = []
    y_AF = []
    y_MAF = []
    y_LPF = []
    y_KF = []

    signal = pd.read_csv("01_filter/Data/example_KalmanFilter_1.csv")

    # 초기화 (첫 번째 측정값을 기반으로 필터 초기값 설정)
    y_initial = signal.y_measure[0]
    
    y_estimate_AF = AverageFilter(y_initial)
    y_estimate_MAF = MovingAverageFilter(y_initial_measure=y_initial)  # window_size는 필요에 따라 조정
    y_estimate_LPF = LowPassFilter(alpha=0.1, y_initial_measure=y_initial)           # alpha도 필요에 따라 조정
    y_estimate_KF = KalmanFilter(y_initial)

    for i, row in signal.iterrows():
        t.append(signal.time[i])

        # Average Filter
        y_estimate_AF.estimate(signal.y_measure[i])
        y_AF.append(y_estimate_AF.y_estimate)

        # Moving Average Filter
        y_estimate_MAF.estimate(signal.y_measure[i])
        y_MAF.append(y_estimate_MAF.y_estimate)

        # Low Pass Filter
        y_estimate_LPF.estimate(signal.y_measure[i])
        y_LPF.append(y_estimate_LPF.y_estimate)

        # Kalman Filter
        y_estimate_KF.estimate(signal.y_measure[i], signal.u[i])
        y_KF.append(y_estimate_KF.x_estimate)

    # 결과 그래프 출력
    plt.figure()
    plt.plot(signal.time, signal.y_measure, 'k.', label="Measure")
    plt.plot(t, y_AF, 'm-', label="Average Filter")
    plt.plot(t, y_MAF, 'b-', label="Moving Average Filter")
    plt.plot(t, y_LPF, 'c-', label="Low Pass Filter")
    plt.plot(t, y_KF, 'r-', label="Kalman Filter")
    plt.xlabel('time (s)')
    plt.ylabel('signal')
    plt.legend(loc="best")
    plt.axis("equal")
    plt.grid(True)
    plt.show()
