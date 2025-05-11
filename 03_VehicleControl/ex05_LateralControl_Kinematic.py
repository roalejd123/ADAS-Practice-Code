import numpy as np
import matplotlib.pyplot as plt

from VehicleModel_Lat import VehicleModel_Lat

class PID_Controller_Kinematic(object):
    def __init__(self, step_time, y_ref, y_current, P_Gain=0.6, D_Gain=0.99, I_Gain=0.2):
        self.y_ref = y_ref
        self.step_time = step_time

        self.Kp = P_Gain
        self.Kd = D_Gain
        self.Ki = I_Gain

        self.prev_error = 0.0
        self.integral_error = 0.0
        self.u = 0.0  # steering angle (delta)

    def ControllerInput(self, y_ref, y_current):
        error = y_ref - y_current
        self.integral_error += error * self.step_time
        derivative = (error - self.prev_error) / self.step_time

        self.u = self.Kp * error + self.Ki * self.integral_error + self.Kd * derivative
        self.prev_error = error
if __name__ == "__main__":
    step_time = 0.1
    simulation_time = 30.0
    Vx = 3.0
    Y_ref = 4.0
    
    time = []
    X_ego = []
    Y_ego = []
    ego_vehicle = VehicleModel_Lat(step_time, Vx)
    controller = PID_Controller_Kinematic(step_time, Y_ref, ego_vehicle.Y)
    for i in range(int(simulation_time/step_time)):
        time.append(step_time*i)
        X_ego.append(ego_vehicle.X)
        Y_ego.append(ego_vehicle.Y)
        controller.ControllerInput(Y_ref, ego_vehicle.Y)
        ego_vehicle.update(controller.u, Vx)

        
    plt.figure(1)
    plt.plot(X_ego, Y_ego,'b-',label = "Position")
    plt.plot([0, X_ego[-1]], [Y_ref, Y_ref], 'k:',label = "Reference")
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend(loc="best")
#    plt.axis("best")
    plt.grid(True)    
    plt.show()


