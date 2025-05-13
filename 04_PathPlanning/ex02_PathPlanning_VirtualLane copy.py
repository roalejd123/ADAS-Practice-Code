import numpy as np
import matplotlib.pyplot as plt
from lane_2 import lane
from ex01_PathPlanning_BothLane import Global2Local, Polyfit, VehicleModel_Lat, PurePursuit

class LaneWidthEstimator(object):
    def __init__(self, Lw_init=3.0):
        self.Lw = Lw_init

    def update(self, coeff_L, coeff_R, isLaneValid_L, isLaneValid_R):
        if all(isLaneValid_L) and all(isLaneValid_R):
            x = np.linspace(0, 5.0, 20)
            y_L = np.polyval(coeff_L, x)
            y_R = np.polyval(coeff_R, x)
            width = np.mean(y_R - y_L)
            self.Lw = 0.9 * self.Lw + 0.1 * width  # 평활 필터



def shift_poly(coeff, shift, direction="+"):
    """
    주어진 다항식 계수(coeff)를 기반으로 일정 간격 평행 이동한 좌표 (x, y)를 반환
    direction이 '+'이면 y값을 shift만큼 위로 이동 (오른쪽 차선 추정), '-'이면 아래로
    """
    x = np.linspace(0, 5.0, 10)
    y = np.polyval(coeff, x)
    if direction == "+":
        y_shifted = y + shift
    else:
        y_shifted = y - shift
    return x, y_shifted

def EitherLane2Path(coeff_L, coeff_R, isLaneValid_L, isLaneValid_R, Lw):
    x = np.linspace(0, 5.0, 20)  # 고정된 로컬 프레임에서 앞쪽 5m

    if all(isLaneValid_L) and all(isLaneValid_R):
        y_L = np.polyval(coeff_L, x)
        y_R = np.polyval(coeff_R, x)
        y_path = (y_L + y_R) / 2.0
        coeff_path = np.polyfit(x, y_path, 3).tolist()

    elif all(isLaneValid_L):
        y_L = np.polyval(coeff_L, x)
        y_R_est = y_L + Lw  # 오른쪽 차선 추정
        y_path = (y_L + y_R_est) / 2.0
        coeff_path = np.polyfit(x, y_path, 3).tolist()

    elif all(isLaneValid_R):
        y_R = np.polyval(coeff_R, x)
        y_L_est = y_R - Lw  # 왼쪽 차선 추정
        y_path = (y_L_est + y_R) / 2.0
        coeff_path = np.polyfit(x, y_path, 3).tolist()

    else:
        coeff_path = [0.0, 0.0, 0.0, 0.0]

    return coeff_path



        
if __name__ == "__main__":
    step_time = 0.1
    simulation_time = 30.0
    Vx = 3.0
    X_lane = np.arange(0.0, 100.0, 0.1)
    Y_lane_L, Y_lane_R, Valid_L, Valid_R = lane(X_lane)
    
    LaneWidth = LaneWidthEstimator()
    ego_vehicle = VehicleModel_Lat(step_time, Vx)
    controller = PurePursuit()
    
    time = []
    X_ego = []
    Y_ego = []
    
    for i in range(int(simulation_time/step_time)):
        time.append(step_time*i)
        X_ego.append(ego_vehicle.X)
        Y_ego.append(ego_vehicle.Y)
        # Lane Info
        X_ref = np.arange(ego_vehicle.X, ego_vehicle.X+5.0, 1.0)
        Y_ref_L, Y_ref_R, isLaneValid_L, isLaneValid_R = lane(X_ref)
        # Global points (front 5 meters from the ego vehicle)
        global_points_L = np.transpose(np.array([X_ref, Y_ref_L])).tolist()
        global_points_R = np.transpose(np.array([X_ref, Y_ref_R])).tolist()
        # Converted to local frame
        local_points_L = Global2Local(global_points_L, ego_vehicle.Yaw, ego_vehicle.X, ego_vehicle.Y)
        local_points_R = Global2Local(global_points_R, ego_vehicle.Yaw, ego_vehicle.X, ego_vehicle.Y)
        # 3th order fitting
        coeff_L = Polyfit(local_points_L, num_order=3)
        coeff_R = Polyfit(local_points_R, num_order=3)
        # Lane to path
        LaneWidth.update(coeff_L, coeff_R, isLaneValid_L, isLaneValid_R)
        coeff_path = EitherLane2Path(coeff_L, coeff_R, isLaneValid_L, isLaneValid_R, LaneWidth.Lw)
        # Controller input
        controller.ControllerInput(coeff_path, Vx)
        ego_vehicle.update(controller.u, Vx)
        
    plt.figure(1, figsize=(13,2))
    plt.plot(X_lane, Y_lane_L,'k--')
    plt.plot(X_lane, Y_lane_R,'k--',label = "Reference")
    plt.plot(X_ego, Y_ego,'b.',label = "Vehicle Position")
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend(loc="best")
#    plt.axis("best")
    plt.grid(True)    
    plt.show()