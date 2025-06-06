from vehicle_model import VehicleModel
import numpy as np
import matplotlib.pyplot as plt

class PID_Controller(object):
    def __init__(self, reference, measure, step_time, P_Gain=0.5, D_Gain=1.2, I_Gain=0.02):
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

if __name__ == "__main__":
    target_y = 0.0
    measure_y = []
    time = []
    step_time = 0.1
    simulation_time = 30
    plant = VehicleModel(step_time, 0.0, 0.99, 0.05)
    controller = PID_Controller(target_y, plant.y_measure[0][0], step_time)

    for i in range(int(simulation_time / step_time)):
        time.append(step_time * i)
        measure_y.append(plant.y_measure[0][0])
        controller.ControllerInput(target_y, plant.y_measure[0][0])
        plant.ControlInput(controller.u)

    plt.figure()
    plt.plot([0, time[-1]], [target_y, target_y], 'k-', label="Reference (y=0)")
    plt.plot(time, measure_y, 'r-', label="Vehicle Position")
    plt.xlabel('Time (s)')
    plt.ylabel('Y Position')
    plt.legend(loc="best")
    plt.grid(True)
    plt.title("PID Controller Result")
    plt.show()
