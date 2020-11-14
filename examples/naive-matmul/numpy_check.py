import numpy as np

tolerance = 1e-3

a = np.loadtxt("matrix.txt")
b = np.loadtxt("result.txt")

assert a.shape == b.shape, "input and output have mismatching shapes"
assert (np.abs((a @ a) - b) < tolerance).all(), "input@input does not match output with tolerance {}".format(tolerance)
