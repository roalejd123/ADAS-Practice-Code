from vehicle_model import VehicleModel
import numpy as np
import matplotlib.pyplot as plt

class PID_Controller(object):
    def __init__(self, reference, measure, step_time, P_Gain=6, D_Gain=5.0, I_Gain=0.1):
        self.Kp = P_Gain
        self.Kd = D_Gain
        self.Ki = I_Gain
        self.prev_error = reference - measure
        self.integral_error = 0.0
        self.dt = step_time
        self.u = 0.0

    def ControllerInput(self, reference, measure):
        error = reference - measure
        self.integral_error += error * self.dt
        d_error = (error - self.prev_error) / self.dt
        self.u = self.Kp * error + self.Ki * self.integral_error + self.Kd * d_error
        self.prev_error = error

class KalmanFilter:
    def __init__(self, y_initial):
        # 초기 상태 [y, v]ᵀ (위치와 속도)
        self.x_estimate = np.array([[y_initial], [0.0]])  # 상태 추정값
        self.A = np.array([[1.0, 0.1], [0.0, 1.0]])        # 시스템 행렬
        self.B = np.array([[0.005], [0.1]])                # 입력 행렬 (dt^2/2, dt)
        self.H = np.array([[1.0, 0.0]])                    # 측정 행렬
        self.P = np.eye(2) * 1.0                           # 오차 공분산 초기값
        self.Q = np.eye(2) * 0.1                          # 프로세스 노이즈
        self.R = np.array([[50]])                        # 측정 노이즈

    def estimate(self, y_measure, u):
        # 예측 단계
        self.x_estimate = self.A @ self.x_estimate + self.B * u
        self.P = self.A @ self.P @ self.A.T + self.Q

        # 업데이트 단계
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        y_tilde = y_measure - self.H @ self.x_estimate  # Innovation
        self.x_estimate += K @ y_tilde
        self.P = (np.eye(2) - K @ self.H) @ self.P

if __name__ == "__main__":
    target_y = 0.0
    measure_y = []
    estimated_y = []
    time = []
    step_time = 0.1
    simulation_time = 30

    plant = VehicleModel(step_time, 0.25, 0.99, 0.05)
    estimator = KalmanFilter(plant.y_measure[0][0])
    controller = PID_Controller(target_y, plant.y_measure[0][0], step_time)

    for i in range(int(simulation_time / step_time)):
        time.append(step_time * i)
        measure_y.append(plant.y_measure[0][0])
        estimated_y.append(estimator.x_estimate[0][0])
        estimator.estimate(plant.y_measure[0][0], controller.u)
        controller.ControllerInput(target_y, estimator.x_estimate[0][0])
        plant.ControlInput(controller.u)

    plt.figure()
    plt.plot([0, time[-1]], [target_y, target_y], 'k-', label="Reference")
    plt.plot(time, measure_y, 'r:', label="Vehicle Position (Measured)")
    plt.plot(time, estimated_y, 'c-', label="Vehicle Position (Kalman Estimate)")
    plt.xlabel('Time (s)')
    plt.ylabel('Y Position')
    plt.legend(loc="best")
    plt.grid(True)
    plt.title("PID Control with Kalman Filter")
    plt.show()
