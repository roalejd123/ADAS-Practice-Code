import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class KalmanFilter:
    def __init__(self, y_Measure_init, step_time=0.001, m=1.0, Q_x=0.05, Q_v=0.1, R=1.0, errorCovariance_init=10.0):
        self.A = np.array([[1.0, step_time], [0.0, 1.0]])
        self.B = np.array([[0.0], [step_time / m]])
        self.C = np.array([[1.0, 0.0]])
        self.Q = np.array([[Q_x, 0.0], [0.0, Q_v]])
        self.R = R
        self.x_estimate = np.array([[y_Measure_init], [0.0]])  # 초기 상태: 위치 + 속도
        self.P_estimate = np.eye(2) * errorCovariance_init     # 초기 공분산 행렬 (2x2)

    def estimate(self, y_measure, input_u):
        # 예측 단계
        x_predict = self.A @ self.x_estimate + self.B * input_u
        P_predict = self.A @ self.P_estimate @ self.A.T + self.Q

        # 칼만 이득 계산
        S = self.C @ P_predict @ self.C.T + self.R
        K = P_predict @ self.C.T @ np.linalg.inv(S)

        # 보정 단계
        y_residual = y_measure - (self.C @ x_predict)
        self.x_estimate = x_predict + K @ y_residual
        self.P_estimate = (np.eye(2) - K @ self.C) @ P_predict

if __name__ == "__main__":
    signal = pd.read_csv("01_filter/Data/example07.csv")
    signal["y_estimate"] = 0.0

    kf = KalmanFilter(signal.y_measure[0])

    for i in range(1, len(signal)):
        kf.estimate(signal.y_measure[i], signal.u[i])
        signal.loc[i, "y_estimate"] = kf.x_estimate[0, 0]  # 위치만 저장

    plt.figure()
    plt.plot(signal.time, signal.y_measure, 'k.', label="Measure")
    plt.plot(signal.time, signal.y_estimate, 'r-', label="Estimate")
    plt.xlabel('time (s)')
    plt.ylabel('signal')
    plt.legend(loc="best")
    plt.axis("equal")
    plt.grid(True)
    plt.show()
