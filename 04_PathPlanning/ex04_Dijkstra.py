import numpy as np
import math
import matplotlib.pyplot as plt
import heapq
from map_1 import map

show_animation = True

class Node:
    def __init__(self, parent=None, position=None):
        self.parent = parent
        self.position = position
        self.f = 0  # Dijkstra에서는 f = cost so far

    def __eq__(self, other):
        return self.position == other.position

    def __lt__(self, other):
        return self.f < other.f

def get_action():
    # dx, dy, cost
    action_set = [
        [0, 1, 1],   # up
        [0, -1, 1],  # down
        [-1, 0, 1],  # left
        [1, 0, 1],   # right
        [-1, -1, math.sqrt(2)], # down-left
        [-1, 1, math.sqrt(2)],  # up-left
        [1, -1, math.sqrt(2)],  # down-right
        [1, 1, math.sqrt(2)]    # up-right
    ]
    return action_set

def collision_check(omap, node):
    x, y = node.position
    for ox, oy in zip(omap[0], omap[1]):
        if x == ox and y == oy:
            return True
    return False

def reconstruct_path(current_node):
    path = []
    while current_node is not None:
        path.append(current_node.position)
        current_node = current_node.parent
    return path[::-1]

def dijkstra(start, goal, map_obstacle):
    start_node = Node(None, start)
    goal_node = Node(None, goal)

    open_list = []
    closed_set = set()

    heapq.heappush(open_list, (start_node.f, start_node))

    while open_list:
        _, current_node = heapq.heappop(open_list)

        if current_node.position == goal_node.position:
            return reconstruct_path(current_node)

        closed_set.add(current_node.position)

        for action in get_action():
            dx, dy, cost = action
            new_pos = (current_node.position[0] + dx, current_node.position[1] + dy)
            new_node = Node(current_node, new_pos)
            new_node.f = current_node.f + cost

            if collision_check(map_obstacle, new_node) or new_node.position in closed_set:
                continue

            skip = False
            for f, n in open_list:
                if n.position == new_node.position and new_node.f >= n.f:
                    skip = True
                    break
            if skip:
                continue

            heapq.heappush(open_list, (new_node.f, new_node))

        if show_animation:
            plt.plot(current_node.position[0], current_node.position[1], 'yo', alpha=0.3)
            if len(closed_set) % 50 == 0:
                plt.pause(0.01)

    return []

def main():
    start, goal, omap = map()

    if show_animation:
        plt.figure(figsize=(8, 8))
        plt.plot(start[0], start[1], 'bs', markersize=7)
        plt.text(start[0], start[1] + 0.5, 'start', fontsize=12)
        plt.plot(goal[0], goal[1], 'rs', markersize=7)
        plt.text(goal[0], goal[1] + 0.5, 'goal', fontsize=12)
        plt.plot(omap[0], omap[1], '.k', markersize=10)
        plt.grid(True)
        plt.axis("equal")
        plt.xlabel("X [m]")
        plt.ylabel("Y [m]")
        plt.title("Dijkstra algorithm", fontsize=20)

    opt_path = dijkstra(start, goal, omap)
    print("Optimal path found!")
    opt_path = np.array(opt_path)

    if show_animation and len(opt_path) > 0:
        plt.plot(opt_path[:, 0], opt_path[:, 1], "m.-")
        plt.show()

if __name__ == "__main__":
    main()
