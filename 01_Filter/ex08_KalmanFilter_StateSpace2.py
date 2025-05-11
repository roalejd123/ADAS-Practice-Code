import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class KalmanFilter:
    def __init__(self, y_init, step_time=0.1, m=10.0, k=100.0, b=2.0,
                 Q_x=0.01, Q_v=0.1, R=5.0, errorCov_init=10.0):
        # 상태 공간 모델 행렬 설정
        self.A = np.array([[1.0, step_time],
                           [-step_time * k / m, 1.0 - step_time * b / m]])
        self.B = np.array([[0.0],
                           [step_time / m]])
        self.C = np.array([[1.0, 0.0]])  # 위치만 측정
        self.Q = np.array([[Q_x, 0.0],
                           [0.0, Q_v]])  # 프로세스 노이즈
        self.R = R  # 측정 노이즈

        # 초기 상태 추정값 (위치, 속도)
        self.x_estimate = np.array([[y_init], [0.0]])
        self.P_estimate = np.eye(2) * errorCov_init  # 초기 오차 공분산

    def estimate(self, y_measure, input_u):
        # 1. 예측 단계
        x_predict = self.A @ self.x_estimate + self.B * input_u
        P_predict = self.A @ self.P_estimate @ self.A.T + self.Q

        # 2. 칼만 이득 계산
        S = self.C @ P_predict @ self.C.T + self.R
        K = P_predict @ self.C.T @ np.linalg.inv(S)

        # 3. 보정 단계
        y_residual = y_measure - (self.C @ x_predict)
        self.x_estimate = x_predict + K @ y_residual
        self.P_estimate = (np.eye(2) - K @ self.C) @ P_predict

if __name__ == "__main__":
    # CSV 파일 로드 (시간, 입력 u, 측정된 위치 y 포함)
    signal = pd.read_csv("01_filter/Data/example08.csv")
    signal["y_estimate"] = 0.0

    # 칼만 필터 초기화 (첫 측정값 기반)
    kf = KalmanFilter(signal.y_measure[0])

    # 전체 신호에 대해 필터 적용
    for i in range(1, len(signal)):
        kf.estimate(signal.y_measure[i], signal.u[i])
        signal.loc[i, "y_estimate"] = kf.x_estimate[0, 0]  # 위치만 저장

    # 결과 시각화
    plt.figure()
    plt.plot(signal.time, signal.y_measure, 'k.', label="Measure")
    plt.plot(signal.time, signal.y_estimate, 'r-', label="Estimate")
    plt.xlabel('time (s)')
    plt.ylabel('position x')
    plt.legend(loc="best")
    plt.axis("equal")
    plt.grid(True)
    plt.show()
