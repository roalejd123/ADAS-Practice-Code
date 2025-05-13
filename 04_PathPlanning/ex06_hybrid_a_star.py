import numpy as np
import math
import matplotlib.pyplot as plt
from map_3 import map

show_animation = True

class Node:
    def __init__(self, parent=None, position=None):
        self.parent = parent
        self.position = position  # [x, y, yaw]
        self.heading = 0.0
        self.f = 0
        self.g = 0
        self.h = 0

def isSamePosition(node_1, node_2, epsilon_position=0.3):
    dx = node_1.position[0] - node_2.position[0]
    dy = node_1.position[1] - node_2.position[1]
    return np.hypot(dx, dy) < epsilon_position

def isSameYaw(node_1, node_2, epsilon_yaw=0.2):
    dyaw = abs(node_1.heading - node_2.heading)
    return dyaw < epsilon_yaw

def get_action(R, Vx, delta_time_step):
    yaw_rate = Vx / R
    distance_travel = Vx * delta_time_step
    action_set = [
        [yaw_rate, delta_time_step, distance_travel],
        [-yaw_rate, delta_time_step, distance_travel],
        [yaw_rate / 2, delta_time_step, distance_travel],
        [-yaw_rate / 2, delta_time_step, distance_travel],
        [0.0, delta_time_step, distance_travel]
    ]
    return action_set

def vehicle_move(position_parent, yaw_rate, delta_time, Vx):
    x, y, yaw = position_parent
    if abs(yaw_rate) > 1e-3:
        R = Vx / yaw_rate
        x_child = x + R * (math.sin(yaw + yaw_rate * delta_time) - math.sin(yaw))
        y_child = y - R * (math.cos(yaw + yaw_rate * delta_time) - math.cos(yaw))
        yaw_child = yaw + yaw_rate * delta_time
    else:
        x_child = x + Vx * delta_time * math.cos(yaw)
        y_child = y + Vx * delta_time * math.sin(yaw)
        yaw_child = yaw

    yaw_child = yaw_child % (2 * math.pi)
    return [x_child, y_child, yaw_child]

def collision_check(position_parent, yaw_rate, delta_time_step, obstacle_list, Vx):
    x, y, _ = vehicle_move(position_parent, yaw_rate, delta_time_step, Vx)
    for obs in obstacle_list:
        ox, oy, r = obs
        if np.hypot(ox - x, oy - y) <= r:
            return True
    return False

def isNotInSearchingSpace(position_child, space):
    x, y = position_child[0], position_child[1]
    if x < space[0] or x > space[1] or y < space[2] or y > space[3]:
        return True
    return False

def heuristic(cur_node, goal_node):
    dx = cur_node.position[0] - goal_node.position[0]
    dy = cur_node.position[1] - goal_node.position[1]
    return np.hypot(dx, dy)

def a_star(start, goal, space, obstacle_list, R, Vx, delta_time_step, weight):
    start_node = Node(None, start)
    start_node.heading = start[2]
    goal_node = Node(None, goal)
    goal_node.heading = goal[2]

    open_list = [start_node]
    closed_list = []

    while open_list:
        cur_node = min(open_list, key=lambda node: node.f)
        open_list.remove(cur_node)
        closed_list.append(cur_node)

        if isSamePosition(cur_node, goal_node, epsilon_position=0.5) and isSameYaw(cur_node, goal_node, epsilon_yaw=0.3):
            path = []
            while cur_node is not None:
                path.append(cur_node.position[:2])
                cur_node = cur_node.parent
            return path[::-1]

        action_set = get_action(R, Vx, delta_time_step)
        for action in action_set:
            yaw_rate, dt, cost = action
            new_pos = vehicle_move(cur_node.position, yaw_rate, dt, Vx)

            if isNotInSearchingSpace(new_pos, space):
                continue
            if collision_check(cur_node.position, yaw_rate, dt, obstacle_list, Vx):
                continue

            child = Node(cur_node, new_pos)
            child.heading = new_pos[2]
            child.g = cur_node.g + cost
            child.h = heuristic(child, goal_node)
            child.f = child.g + weight * child.h

            if any(isSamePosition(child, c) and isSameYaw(child, c) for c in closed_list):
                continue
            if any(isSamePosition(child, o) and isSameYaw(child, o) and child.g >= o.g for o in open_list):
                continue

            open_list.append(child)

        if show_animation:
            plt.plot(cur_node.position[0], cur_node.position[1], 'yo', alpha=0.3)
            if len(closed_list) % 100 == 0:
                plt.pause(0.01)

    return []

def main():
    start, goal, obstacle_list, space = map()

    if show_animation:
        theta_plot = np.linspace(0, 1, 101) * np.pi * 2
        plt.figure(figsize=(8, 8))
        plt.plot(start[0], start[1], 'bs', markersize=7)
        plt.text(start[0], start[1] + 0.5, 'start', fontsize=12)
        plt.plot(goal[0], goal[1], 'rs', markersize=7)
        plt.text(goal[0], goal[1] + 0.5, 'goal', fontsize=12)
        for obs in obstacle_list:
            ox, oy, r = obs
            x_obstacle = ox + r * np.cos(theta_plot)
            y_obstacle = oy + r * np.sin(theta_plot)
            plt.plot(x_obstacle, y_obstacle, 'k-')
        plt.axis(space)
        plt.grid(True)
        plt.xlabel("X [m]"), plt.ylabel("Y [m]")
        plt.title("Hybrid A* Algorithm", fontsize=20)

    opt_path = a_star(start, goal, space, obstacle_list, R=5.0, Vx=2.0, delta_time_step=0.5, weight=1.1)
    if opt_path:
        print("Optimal path found!")
        opt_path = np.array(opt_path)
        if show_animation:
            plt.plot(opt_path[:, 0], opt_path[:, 1], "m.-")
            plt.show()
    else:
        print("No path found.")

if __name__ == "__main__":
    main()
