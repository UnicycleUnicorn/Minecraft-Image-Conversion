from scipy.spatial import KDTree
import numpy as np
for i in range(10000):
    # generate some random 3D points
    points = np.random.rand(1000, 3)

    # create a KDTree from the points
    tree = KDTree(points)

    # pick a random query point
    query_point = np.random.rand(1, 3)

    # find the nearest neighbor to the query point using brute force
    bf_dists = np.linalg.norm(points - query_point, axis=1)
    bf_nearest = np.argmin(bf_dists)

    # find the nearest neighbor to the query point using the KDTree
    kd_dists, kd_indices = tree.query(query_point, k=1)
    kd_nearest = kd_indices[0]

    # compare the results
    if bf_nearest != kd_nearest:
        print(f"Brute-force nearest neighbor index: {bf_nearest}")
        print(f"KDTree nearest neighbor index: {kd_nearest}")