import numpy as np
import matplotlib.pyplot as plt

from VehicleModel_Long import VehicleModel_Long

class PID_Controller_ConstantSpace(object):
    def __init__(self, step_time, target_x, ego_x, constantSpace=30.0, P_Gain=1, D_Gain=1, I_Gain=0.3):
        self.space = constantSpace
        self.step_time = step_time

        self.Kp = P_Gain
        self.Kd = D_Gain
        self.Ki = I_Gain

        self.prev_error = 0.0
        self.integral_error = 0.0
        self.u = 0.0

    def ControllerInput(self, target_x, ego_x):
        error = (target_x - ego_x) - self.space
        self.integral_error += error * self.step_time
        derivative = (error - self.prev_error) / self.step_time

        self.u = self.Kp * error + self.Ki * self.integral_error + self.Kd * derivative
        self.prev_error = error


if __name__ == "__main__":
    
    step_time = 0.1
    simulation_time = 50.0
    m = 500.0
    
    vx_ego = []
    vx_target = []
    x_space = []
    time = []
    target_vehicle = VehicleModel_Long(step_time, m, 0.0, 30.0, 10.0)  # 앞차는 가속 없음, 30m/s로 이동
    ego_vehicle = VehicleModel_Long(step_time, m, 0.5, 0.0, 10.0)      # 뒷차 초기속도 10m/s
    controller = PID_Controller_ConstantSpace(step_time, target_vehicle.x, ego_vehicle.x)

    for i in range(int(simulation_time/step_time)):
        time.append(step_time * i)
        vx_ego.append(ego_vehicle.vx)
        vx_target.append(target_vehicle.vx)
        x_space.append(target_vehicle.x - ego_vehicle.x)

        controller.ControllerInput(target_vehicle.x, ego_vehicle.x)
        ego_vehicle.update(controller.u)
        target_vehicle.update(0.0)  # 앞차는 계속 같은 속도 유지
        
    plt.figure(1)
    plt.plot(time, vx_ego, 'r-', label="ego_vx [m/s]")
    plt.plot(time, vx_target, 'b-', label="target_vx [m/s]")
    plt.xlabel('time [s]')
    plt.ylabel('Vx')
    plt.legend(loc="best")
    plt.grid(True)
    
    plt.figure(2)
    plt.plot([0, time[-1]], [controller.space, controller.space], 'k--', label="reference space [30m]")
    plt.plot(time, x_space, 'b-', label="actual space [m]")
    plt.xlabel('time [s]')
    plt.ylabel('Distance between cars [m]')
    plt.legend(loc="best")
    plt.grid(True)
    
    plt.show()
