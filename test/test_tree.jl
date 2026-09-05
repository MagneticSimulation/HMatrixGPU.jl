using Random
using HMatrixGPU
using Test
Random.seed!(10)

N = 1000;
coordinates = rand(3, N)  # N points in 3D space
coordinates[3, :] .= 0  # set the z-coordinate to 0

cluster = ClusterTree(coordinates; max_points_per_leaf=10)

s = info(cluster)

@test s[:max_points] <= 10
# box-midpoint splits guarantee at least one point per side; the balance of
# leaf sizes is statistical, so the hard invariant is "no empty leaf"
@test s[:min_points] >= 1

@testset "degenerate cluster trees" begin
    # all-identical points: the mean split used to leave a side empty and
    # crash on an empty bounding box; now degenerate nodes become leaves
    pts_dup = repeat(rand(3, 5), 1, 4)
    t = ClusterTree(pts_dup; max_points_per_leaf=4)
    @test sort(t.index_map) == collect(1:20)

    pts_mix = hcat(repeat(rand(3, 1), 1, 19), rand(3, 1) .+ 10)
    t = ClusterTree(pts_mix; max_points_per_leaf=4)
    @test sort(t.index_map) == collect(1:20)

    # duplicates mixed into a regular ring must not break the tree
    N = 200
    ring = [[sin(2π*i/N), cos(2π*i/N), 0.0] for i in 1:N]
    pts_ring = hcat(reduce(hcat, ring), repeat(ring[1], 1, 5))
    t = ClusterTree(pts_ring; max_points_per_leaf=64)
    @test sort(t.index_map) == collect(1:205)
    @test t.root.end_idx - t.root.start_idx == 205
end
