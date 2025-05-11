import numpy as np
import matplotlib.pyplot as plt

from VehicleModel_Lat import VehicleModel_Lat
from ex06_GlobalFrame2LocalFrame import Global2Local, PolynomialFitting, PolynomialValue

class StanleyMethod:
    def __init__(self, step_time, coeff, velocity, K=1.0):
        self.dt = step_time
        self.v = velocity
        self.K = K
        self.u = 0.0

    def ControllerInput(self, coeff, velocity):
        x = 0.0
        path_slope = 3 * coeff[0][0]*x**2 + 2*coeff[1][0]*x + coeff[2][0]
        path_yaw = np.arctan(path_slope)
        heading_error = path_yaw
        cte = coeff[3][0]
        self.u = heading_error + np.arctan2(self.K * cte, velocity + 1e-5)

if __name__ == "__main__":
    step_time = 0.1
    simulation_time = 30.0
    Vx = 3.0
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
    controller = StanleyMethod(step_time, polynomialfit.coeff, Vx)

    for i in range(int(simulation_time / step_time)):
        time.append(step_time * i)
        X_ego.append(ego_vehicle.X)
        Y_ego.append(ego_vehicle.Y)

        X_ref_segment = np.arange(ego_vehicle.X, ego_vehicle.X + 5.0, 1.0)
        Y_ref_segment = 2.0 - 2 * np.cos(X_ref_segment / 10)
        Points_ref = np.column_stack((X_ref_segment, Y_ref_segment))

        frameconverter.convert(Points_ref, ego_vehicle.Yaw, ego_vehicle.X, ego_vehicle.Y)
        polynomialfit.fit(frameconverter.LocalPoints)
        polynomialvalue.calculate(polynomialfit.coeff, x_local)

        controller.ControllerInput(polynomialfit.coeff, Vx)
        ego_vehicle.update(controller.u, Vx)
        print(f"[Step {i}] coeff.T = {polynomialfit.coeff.T}")
        print(f"[Step {i}] LocalPoints =\n{frameconverter.LocalPoints}")

    plt.figure(1)
    plt.plot(X_ref, Y_ref, 'k-', label="Reference")
    plt.plot(X_ego, Y_ego, 'b-', label="Position")
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.grid(True)
    plt.axis("equal")
    plt.title("Stanley Control (Local Frame)")
    plt.show()
