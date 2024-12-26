from collections import defaultdict
import numpy as np

def get_hough_line(points, interval=1):
    accumulator = defaultdict(int)
    for point in points:
        for angle in range(0, 180, interval):
            theta = np.deg2rad(angle)  # 角度转弧度
            rho = point[0] * np.cos(theta) + point[1] * np.sin(theta)
            rho = round(rho, 5)
            theta = round(theta, 5)
            accumulator[(rho, theta)] += 1  # 存储(ρ, θ)和对应的投票数
    
    best_votes = 0
    best_rho = None
    best_theta = None
    for (rho, theta), votes in accumulator.items():
        if votes > best_votes:
            best_votes = votes
            best_rho = rho
            best_theta = theta
    return best_rho, best_theta, best_votes


if __name__ == "__main__":
    # y = x - 1
    points = [(1, 0), (2, 1), (3, 2)]
    rho, theta, votes = get_hough_line(points, interval=1)
    print(f"Votes: {votes}, rho: {rho}, theta: {theta}")
    if rho is not None and theta is not None:
        print(f"Formula of line is: {rho} = {np.cos(theta)} * x + {np.sin(theta)} * y") 

        