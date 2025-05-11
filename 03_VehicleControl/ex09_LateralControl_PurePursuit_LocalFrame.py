import numpy as np
import matplotlib.pyplot as plt

from VehicleModel_Lat import VehicleModel_Lat
from ex06_GlobalFrame2LocalFrame import Global2Local, PolynomialFitting, PolynomialValue

class PurePursuit:
    def __init__(self, step_time, coeff, velocity, L=2.9, lookahead_distance=2.0):
        self.dt = step_time
        self.L = L
        self.v = velocity
        self.lookahead_distance = lookahead_distance
        self.u = 0.0

    def ControllerInput(self, coeff, velocity):
        x_ld = self.lookahead_distance
        x_row = np.array([x_ld**3, x_ld**2, x_ld, 1])
        y_ld = np.dot(coeff.T, x_row)
        self.u = np.arctan2(2 * self.L * y_ld, x_ld**2)

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
    controller = PurePursuit(step_time, polynomialfit.coeff, Vx)

    for i in range(int(simulation_time / step_time)):
        time.append(step_time * i)
        X_ego.append(float(ego_vehicle.X))   # ← 핵심 수정
        Y_ego.append(float(ego_vehicle.Y))   

        X_ref_convert = np.arange(ego_vehicle.X, ego_vehicle.X + 5.0, 1.0)
        Y_ref_convert = 2.0 - 2 * np.cos(X_ref_convert / 10)
        Points_ref = np.column_stack((X_ref_convert, Y_ref_convert))
       

        print(f"[Step {i}] Yaw = {ego_vehicle.Yaw}")
        print(f"[Step {i}] Y =\n{ego_vehicle.Y}")

        frameconverter.convert(Points_ref, ego_vehicle.Yaw, ego_vehicle.X, ego_vehicle.Y)
        polynomialfit.fit(frameconverter.LocalPoints)
        polynomialvalue.calculate(polynomialfit.coeff, x_local)

        controller.ControllerInput(polynomialfit.coeff, Vx)
        ego_vehicle.update(controller.u, Vx)


    


    plt.figure(1)
    plt.plot(X_ref, Y_ref, 'k-', label="Reference")
    plt.plot(X_ego, Y_ego, 'b-', label="Position")
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.grid(True)
    plt.axis("equal")
    plt.title("Pure Pursuit (Local Frame)")
    plt.show()
