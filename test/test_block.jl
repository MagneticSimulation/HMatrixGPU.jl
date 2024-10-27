using Random
using LinearAlgebra
using HMatrixGPU
using Test
Random.seed!(10)

N = 32;
K = rand(N, N) 
coordinates = rand(3, N)  # N points in 3D space
coordinates[2:3, :] .= 0 

source_tree = ClusterTree(coordinates, max_points_per_leaf=4)
target_tree = ClusterTree(coordinates, max_points_per_leaf=8)


block_tree = BlockTree(K, target_tree, source_tree, eta=1.5)


# Test if the root node exists
@testset "BlockTree Tests" begin
    @test block_tree.root != nothing
    @test block_tree.root.left != nothing && block_tree.root.right != nothing

    # Check conditions for leaf nodes
    function check_leaf_node_labels(node::HMatrixGPU.BlockTreeNode)
        if node.is_dense || node.is_admissible
            @test node.left == nothing && node.right == nothing
        elseif node.left != nothing && node.right != nothing
            check_leaf_node_labels(node.left)
            check_leaf_node_labels(node.right)
        end
    end
    check_leaf_node_labels(block_tree.root)

    # Test admissible condition
    block_tree_strict = BlockTree(K, target_tree, source_tree, eta=0.5)
    function check_admissible_conditions(node::HMatrixGPU.BlockTreeNode, eta)
        if node.is_admissible
            dist = norm(node.target_tree.center .- node.source_tree.center)
            radius_sum = node.target_tree.radius + node.source_tree.radius
            @test dist > eta * radius_sum
        elseif node.left != nothing && node.right != nothing
            check_admissible_conditions(node.left, eta)
            check_admissible_conditions(node.right, eta)
        end
    end
    check_admissible_conditions(block_tree_strict.root, 0.5)

    # Test depth of BlockTree
    function tree_depth(node::HMatrixGPU.BlockTreeNode)
        if node.left == nothing && node.right == nothing
            return 1
        else
            left_depth = node.left != nothing ? tree_depth(node.left) : 0
            right_depth = node.right != nothing ? tree_depth(node.right) : 0
            return 1 + max(left_depth, right_depth)
        end
    end
    depth = tree_depth(block_tree.root)
    @test depth > 0

    # Count total number of nodes
    function count_nodes(node)
        if node == nothing
            return 0
        else
            return 1 + count_nodes(node.left) + count_nodes(node.right)
        end
    end
    node_count = count_nodes(block_tree.root)
    @test node_count > 0
end
