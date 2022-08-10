from cmath import exp
import sys
from typing import final
import numpy as np
import spkmeansmodule as spk
MAT_OPS = ("wam", "ddg", "lnorm", "jacobi")

def main():
    try:
        k, goal, file_path = receive_input()
    except(AssertionError):
        print("Invalid Input")
        return
    try:
        obs = np.genfromtxt(file_path, delimiter= ',')
        n, d = obs.shape[0], obs.shape[1]
        assert(k < n)
    except(AssertionError):
        print("Invalid Input")
        return
    except:
        print("An Error Has Occurred")
        return
    if goal in MAT_OPS:
        result = spk.apply_mat_ops(goal, n, d, obs.tolist())
        try:
            assert(result != None)
            print_mat(result)
        except(AssertionError):
            print("An Error Has Occurred")
            return
    else:
        lnorm = spk.apply_mat_ops("lnorm", n, d, obs.tolist())
        try:
            assert(lnorm != None)
        except(AssertionError):
            print("An Error Has Occurred")
            return
        jacobi = spk.apply_mat_ops("jacobi", n, d, lnorm)
        try:
            assert(jacobi != None)
        except(AssertionError):
            print("An Error Has Occurred")
            return
        T = spk.apply_kmeans_prep(n, k, jacobi)
        try:
            assert(T != None)
        except(AssertionError):
            print("An Error Has Occurred")
            return
        try:
            k = len(T[0])
            initial_centroids, indices = kmeanspp(k, T)
            print(",".join([str(elem) for elem in indices]))
            final_centroids = spk.apply_kmeans(len(T), len(T[0]), k, 300, 0, initial_centroids.tolist(), T)
            assert(final_centroids != None)
            print_mat(final_centroids)
            return
        except(AssertionError):
            print("An Error Has Occurred")
            return
    
def receive_input():
    assert len(sys.argv) == 4
    try:
        k = int(sys.argv[1])
    except:
        assert 1 == 0
    assert((sys.argv[2] in MAT_OPS) or (sys.argv[2] == "spk"))
    return k, sys.argv[2], sys.argv[3]

'''
The function finds the distance of the closest centroid to x.
The distance is measured using the euclidean distance.
The function assumes the dimension of the centroids and of x is the same.
'''
def find_closest_distance(x, centroids):
    minimal_distance = sum((x-centroids[0]) ** 2)
    for i in range(1, len(centroids)):
        minimal_distance = min(minimal_distance, sum((x-centroids[i]) ** 2))
    return minimal_distance


def kmeanspp(k, obs):
    np.random.seed(0)
    indices = [np.random.choice(range(len(obs)))]
    centroids = np.array(obs[indices[0]])
    for i in range(1, k):
        distances = np.array([find_closest_distance(obs[j], centroids) for j in range(len(obs))])
        s = sum(distances)
        probs = distances / s
        indices.append(np.random.choice(range(len(obs)), p=probs))
        centroids = np.append(centroids, np.array(obs[indices[-1]]), axis = 0)
    return centroids, indices

def print_mat(mat):
    for i in range(len(mat)):
        print(','.join(["%.4f" % elem for elem in mat[i]]))

if __name__ == "__main__":
    main()