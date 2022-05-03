import numpy as np
import os


def compute_euc(airportPosition, x_vals):
    distances = np.sqrt(np.sum(np.asarray(airportPosition - x_vals) ** 2, axis=1))
    keep = np.where(distances < 1, 1, 0).sum()
    return keep

if __name__ == '__main__':
    pointsList = [[1, 2], [10, 15]]
    a = [3, 4]
    dis = compute_euc(np.array(a), np.array(pointsList))
    keep = np.where(dis < 1, 1, 0).sum()
    print(keep)

