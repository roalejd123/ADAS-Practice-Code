from vehicle_model import VehicleModel
import numpy as np
import matplotlib.pyplot as plt

class P_Controller(object):
    def __init__(self, P_Gain=0.3):
        self.Kp = P_Gain
        self.u = 0.0  # 제어 입력 초기화

    def ControllerInput(self, reference, measure):
        error = reference - measure
        self.u = self.Kp * error  # 비례 제어

if __name__ == "__main__":
    target_y = 0.0
    measure_y = []
    time = []
    step_time = 0.1
    simulation_time = 30

    # VehicleModel(step_time, 초기속도, 감쇠계수?, 초기위치)
    plant = VehicleModel(step_time, 0.0, 0.99, 0.1)
    controller = P_Controller(P_Gain=1)  # Gain은 튜닝 가능

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
    plt.title("P Controller Result")
    plt.show()
