from cmath import exp
import sys
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
        except(AssertionError):
            print("An Error Has Occurred")
            return
    else:
        lnorm = spk.apply_mat_ops("lnorm", n, d, obs)
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
        k = len(T[0])
    
def receive_input():
    assert len(sys.argv) == 4
    try:
        k = int(sys.argv[1])
    except:
        assert 1 == 0
    assert((sys.argv[2] in MAT_OPS) or (sys.argv[2] == "spk"))
    return k, sys.argv[2], sys.argv[3]

def kmeanspp(k, obs):
    np.random.seed(0)