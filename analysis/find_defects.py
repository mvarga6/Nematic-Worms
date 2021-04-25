import argparse
from multiprocessing import Pool, cpu_count
import math
from numpy.core.fromnumeric import size

from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, AffinityPropagation
import warnings
from sklearn.exceptions import ConvergenceWarning

from afa.io import _read_xyz_header


DELTA = math.pi / 4


def theta_diff(t1, t2):
    dt = t2 - t1
    if dt >= math.pi / 2.:
        dt -= math.pi
    elif dt <= -math.pi / 2.:
        dt += math.pi
    return dt

def burgers_circuit(i, j, theta, loop):
    circuit_angle = 0
    di, dj = loop[0]
    t1 = theta[i + di, j + dj]
    for di,dj in loop[1:]:
        t2 = theta[i + di, j + dj]
        circuit_angle += theta_diff(t1, t2)
        t1 = t2
    return circuit_angle


LOOP_A = [
    [0, 0],
    [1, 0],
    [1, 1],
    [0, 1],
    [0, 0],
]

LOOP_B = [
    [1, 0],
    [1, 1],
    [0, 1],
    [-1, 1],
    [-1, 0],
    [-1, -1],
    [0, -1],
    [1, -1],
    [1, 0],
]


def read_vector_field(filename):
    frames = []
    with open(filename, "r") as f:
        count = 0
        while True:
            line = f.readline()
            if line in ["", None]:
                break

            count += 1
            num_sites = int(line)
            comment = f.readline()
            R = np.zeros((num_sites, 3))
            V = np.zeros((num_sites, 3))
            for i in range(num_sites):
                data = [float(x) for x in f.readline().split()]
                R[i] = data[:3]
                V[i] = data[3:]

            origin = R.min(axis=0)
            frames.append({
                "index": count,
                "num_sites": num_sites,
                "positions": R,
                "director": V,
                "origin": origin,
                "box_size": R.max(axis=0) - origin,
            })
    return frames


def compute_theta(frame, cellsize):
    num_sites = frame["num_sites"]
    R = frame["positions"]
    V = frame["director"]
    origin = frame["origin"]
    box_size = frame["box_size"]
    frame["cellsize"] = cellsize
    box_shape = tuple(np.ceil((box_size + 0.0001) / cellsize).astype(np.int))
    theta = np.zeros(box_shape[:2])
    for i in range(num_sites):
        r, v = R[i], V[i]
        cell = np.floor((r - origin) / cellsize).astype(np.int)
        theta[cell[0], cell[1]] = math.atan2(v[1], v[0])
    frame["theta"] = theta
    return frame



def find_defects(frame):
    theta = frame["theta"]
    cellsize = frame["cellsize"]
    origin = frame["origin"]
    counts = np.zeros_like(theta)
    charge = {}

    for i in range(1, theta.shape[0] - 1):
        for j in range(1, theta.shape[1] - 1):
            angle_circuit_a = burgers_circuit(i, j, theta, LOOP_A)
            angle_circuit_b = burgers_circuit(i, j, theta, LOOP_B)

            if angle_circuit_a > math.pi - DELTA or angle_circuit_a < -math.pi + DELTA:
                for di, dj in LOOP_A:
                    counts[i + di, j + dj] += 1
                    charge.setdefault((i + di, j + dj), []).append(angle_circuit_a / (2*math.pi))

            if angle_circuit_b > math.pi - DELTA or angle_circuit_b < -math.pi + DELTA:
                counts[i, j] += 1
                charge.setdefault((i, j), []).append(angle_circuit_b / (2*math.pi))

    defects = []
    if counts.sum() > 0:
        charge = {k: np.mean(v) for k,v in charge.items()}

        X = []
        a,b = np.where(counts > 0)
        for x,y in zip(a,b):
            X.append([x,y])
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            clustering = AffinityPropagation(random_state=None).fit(X)
            for i,j in clustering.cluster_centers_:
                defects.append(("A" if charge[i,j] > 0 else "B", (i*cellsize) + origin[0], (j*cellsize) + origin[1], 0, charge[i,j]))
    return defects


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze XYZ files of active filament simulation.")
    parser.add_argument("-i", "--input", type=str, required=True, help="A directory field sequence XYZV file")
    parser.add_argument("-o", "--output", type=str, default="defects.xyz")
    parser.add_argument("-np", "--num_processes", type=int, default=cpu_count() + 1)
    parser.add_argument("-c", "--cellsize", type=float, required=True)
    args = parser.parse_args()

    print(f"Reading {args.input} ...")
    with open(args.output, "w+") as out:
        for frame in tqdm(read_vector_field(args.input), desc=" Progress"):
            compute_theta(frame, args.cellsize)
            defects = find_defects(frame)
            if len(defects) > 0:
                out.write(f'{len(defects)}\n')
                out.write(f'Defects detected\n')
                for c,x,y,z,q in defects:
                    out.write("%c %f %f %f %f\n" % (c, x, y, z, q))
            else:
                origin = frame["origin"]
                out.write(f'2\n')
                out.write(f'No defects detected\n')
                out.write(f'A {origin[0]} {origin[1]} 0 +0.5\n')
                out.write(f'B {origin[0]} {origin[1]} 0 -0.5\n')
