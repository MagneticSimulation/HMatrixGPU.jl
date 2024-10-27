using Random
using HMatrixGPU
using Test
Random.seed!(10)

N = 1000;
coordinates = rand(3, N)  # N points in 3D space
coordinates[3, :] .= 0  # set the z-coordinate to 0

tree = ClusterTree(coordinates, max_points_per_leaf=10)

s = info(tree)

@test s[:max_points] <= 10
@test s[:min_points] >= 3