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

# ---------------------------------------------------------------------------
# point-set dual form: a d x N matrix passes through by reference, a vector of
# points is stacked once into the same d x N layout (each element a column)
# ---------------------------------------------------------------------------
@testset "point-set dual form" begin
    pts_m = rand(3, 50)
    pts_v = [(pts_m[1, i], pts_m[2, i], pts_m[3, i]) for i in 1:50]
    t_m = ClusterTree(pts_m; max_points_per_leaf=8)
    t_v = ClusterTree(pts_v; max_points_per_leaf=8)
    @test t_m.coordinates === pts_m            # matrix input: stored by reference
    @test t_v.coordinates == pts_m             # stacked once, values as given
    @test t_m.index_map == t_v.index_map       # identical trees
    @test t_m.dims == 1 && t_v.dims == 1

    # vector-of-vectors form works the same way, and dims is carried in the tree
    t_vec = ClusterTree([[pts_m[1, i], pts_m[2, i], pts_m[3, i]] for i in 1:50];
                        max_points_per_leaf=8)
    @test t_vec.coordinates == pts_m
    t3 = ClusterTree(pts_v; max_points_per_leaf=8, dims=3)
    @test t3.dims == 3
    @test t3.coordinates == pts_m
end

# transposed (N x d) input warns about the orientation — this is the only
# transposed input in the whole suite (maxlog=1 would swallow a second log)
@test_logs (:warn, r"possible transposed input") ClusterTree(rand(50, 3))
