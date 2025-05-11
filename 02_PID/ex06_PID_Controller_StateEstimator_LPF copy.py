from vehicle_model import VehicleModel
import numpy as np
import matplotlib.pyplot as plt

class PID_Controller(object):
    def __init__(self, reference, measure, step_time, P_Gain=0.5, D_Gain=1.0, I_Gain=0.05):
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

class LowPassFilter:
    def __init__(self, y_initial_measure, alpha=0.9):
        self.y_estimate = y_initial_measure  # 초기 추정값
        self.alpha = alpha                   # 필터 계수 (0.9는 과거 값에 더 비중)

    def estimate(self, y_measure):
        self.y_estimate = self.alpha * self.y_estimate + (1 - self.alpha) * y_measure

if __name__ == "__main__":
    target_y = 0.0
    measure_y = []
    estimated_y = []
    time = []
    step_time = 0.1
    simulation_time = 30

    plant = VehicleModel(step_time, 0.25, 0.99, 0.05)
    estimator = LowPassFilter(plant.y_measure[0][0])
    controller = PID_Controller(target_y, plant.y_measure[0][0], step_time)

    for i in range(int(simulation_time / step_time)):
        time.append(step_time * i)
        measure_y.append(plant.y_measure[0][0])
        estimated_y.append(estimator.y_estimate)
        estimator.estimate(plant.y_measure[0][0])
        controller.ControllerInput(target_y, estimator.y_estimate)
        plant.ControlInput(controller.u)

    plt.figure()
    plt.plot([0, time[-1]], [target_y, target_y], 'k-', label="Reference")
    plt.plot(time, measure_y, 'r-', label="Vehicle Position (Measured)")
    plt.plot(time, estimated_y, 'c-', label="Vehicle Position (Estimated, LPF)")
    plt.xlabel('Time (s)')
    plt.ylabel('Y Position')
    plt.legend(loc="best")
    plt.axis("equal")
    plt.grid(True)
    plt.title("PID Control with Low Pass Filter")
    plt.show()
