import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class MovingAverageFilter:
    def __init__(self, y_initial_measure, num_average=5):
        self.y_estimate = y_initial_measure  # 초기 추정값
        self.num_average = num_average       # 평균 낼 개수 n
        self.buffer = [y_initial_measure]    # 최근 측정값을 저장할 버퍼

    def estimate(self, y_measure):
        self.buffer.append(y_measure)

        if len(self.buffer) <= self.num_average:
            # 충분한 데이터가 쌓이기 전까지는 단순 평균 사용
            self.y_estimate = np.mean(self.buffer)
        else:
            x_k = y_measure
            x_k_n = self.buffer[-(self.num_average + 1)]
            self.y_estimate = self.y_estimate + (1 / self.num_average) * (x_k - x_k_n)

    # buffer에 계속 값을 추가하되, 오래된 값을 남겨야 x_k-n 참조 가능

if __name__ == "__main__":
    #signal = pd.read_csv("week_01_filter/Data/example_Filter_1.csv")      
    signal = pd.read_csv("01_filter/Data/example_Filter_2.csv")

    signal["y_estimate"] = 0.0  # 추정값 컬럼 초기화

    y_filter = MovingAverageFilter(signal.y_measure[0], num_average=5)
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
