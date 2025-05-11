import numpy as np
import matplotlib.pyplot as plt

from VehicleModel_Lat import VehicleModel_Lat
from ex06_GlobalFrame2LocalFrame import Global2Local, PolynomialFitting, PolynomialValue

class PD_Controller_Kinematic(object):
    def __init__(self, step_time, y_ref, y_current, P_Gain=100, D_Gain=0.1):
        self.step_time = step_time
        self.Kp = P_Gain
        self.Kd = D_Gain
        self.prev_error = 0.0
        self.u = 0.0

    def ControllerInput(self, y_ref, y_current):
        error = y_ref - y_current
        derivative = (error - self.prev_error) / self.step_time
        self.u = self.Kp * error + self.Kd * derivative
        self.prev_error = error

if __name__ == "__main__":
    step_time = 0.1
    simulation_time = 30.0
    Vx = 3.0
    L = 2.9
    X_ref = np.arange(0.0, 100.0, 0.1)
    Y_ref = 2.0 - 2 * np.cos(X_ref / 10)
    num_degree = 3
    num_point = 5
    x_local = np.arange(0.0, 10.0, 0.5)

    time = []
    X_ego = []
    Y_ego = []

    ego_vehicle = VehicleModel_Lat(step_time, Vx)
    frameconverter = Global2Local(num_point)
    polynomialfit = PolynomialFitting(num_degree, num_point)
    polynomialvalue = PolynomialValue(num_degree, len(x_local))
    controller = PD_Controller_Kinematic(step_time, 0.0, 0.0, P_Gain=100, D_Gain=0.1)

    for i in range(int(simulation_time / step_time)):
        time.append(step_time * i)
        X_ego.append(float(ego_vehicle.X))
        Y_ego.append(float(ego_vehicle.Y))

        X_ref_convert = np.arange(ego_vehicle.X, ego_vehicle.X + 5.0, 1.0)
        Y_ref_convert = 2.0 - 2 * np.cos(X_ref_convert / 10)
        Points_ref = np.column_stack((X_ref_convert, Y_ref_convert))

        frameconverter.convert(Points_ref, ego_vehicle.Yaw, ego_vehicle.X, ego_vehicle.Y)
        polynomialfit.fit(frameconverter.LocalPoints)
        polynomialvalue.calculate(polynomialfit.coeff, x_local)

        lookahead_index = 8
        x_ld = x_local[lookahead_index]
        lookahead_y = polynomialvalue.y[lookahead_index][0]

        # PD 제어
        controller.ControllerInput(lookahead_y, 0.0)
        delta_fb = controller.u

        # 피드포워드 (곡률 기반)
        a, b, c, d = polynomialfit.coeff.flatten()
        y_prime = 3*a*x_ld**2 + 2*b*x_ld + c
        y_double_prime = 6*a*x_ld + 2*b
        kappa = y_double_prime / (1 + y_prime**2)**1.5
        delta_ff = np.arctan(L * kappa)

        # 최종 조향각
        delta_total = delta_fb + delta_ff
        ego_vehicle.update(delta_total, Vx)

    plt.figure(1)
    plt.plot(X_ref, Y_ref, 'k-', label="Reference")
    plt.plot(X_ego, Y_ego, 'b-', label="Position")
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.grid(True)
    plt.title("PD + Feedforward Control")
    plt.axis("equal")
    plt.show()
