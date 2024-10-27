using LinearAlgebra

# We translate and simplify the python code at https://tbenthompson.com/book/tdes/hmatrix.html.

# Define the ClusterNode struct
struct ClusterNode
    is_leaf::Bool             # Indicates if this node is a leaf
    start_idx::Int            # Start index of points in this cluster
    end_idx::Int              # End index of points in this cluster
    center::Vector{Float64}   # Center of the cluster
    radius::Float64           # Radius of the cluster
    left::Union{Nothing, ClusterNode}  # Left child node
    right::Union{Nothing, ClusterNode} # Right child node
end

# Define the ClusterTree struct
struct ClusterTree
    index_map::Vector{Int}    # Ordered indices of points in the cluster tree
    root::ClusterNode         # Root node of the cluster tree
end

"""
    ClusterTree(coordinates; max_points_per_leaf::Int=32) -> ClusterTree

Builds a hierarchical clustering tree from a set of points represented by `coordinates`.

# Arguments
- `coordinates::Matrix{Float64}`: A 2D matrix where each column represents a point in space.
- `max_points_per_leaf::Int`: The maximum number of points per leaf node. If the number of points in a node is greater than this value, 
the node is split along its longest axis to create child nodes. If the number of points in a node is less than or equal to this value, 
the node is considered a leaf node and no further splitting is performed. Defaults to 32.

# Returns
A `ClusterTree` struct containing:
  - `index_map`: A vector of point indices reordered to reflect the cluster structure.
  - `root_node`: The root `ClusterNode` representing the entire hierarchical cluster tree.

# Example
```julia
coordinates = rand(3, 100)  # 100 points in 3D space
tree = ClusterTree(coordinates, max_points_per_leaf=5)
```
"""
function ClusterTree(coordinates; max_points_per_leaf::Int=32)
    # Initialize index_map with a sequence of point indices
    index_map = collect(1:size(coordinates, 2))
    
    # Recursively build the tree starting from the root node with all points
    root_node = build_tree_node(coordinates, max_points_per_leaf, index_map, 1, size(coordinates, 2) + 1)
    
    return ClusterTree(index_map, root_node)
end

# Recursive function to build nodes within the tree
function build_tree_node(coordinates::Matrix{Float64}, max_points_per_leaf::Int, index_map::Vector{Int}, start_idx::Int, end_idx::Int)

    # View the portion of index_map corresponding to the current node
    idx_view = view(index_map, start_idx:end_idx-1)
    points = coordinates[:, idx_view]

    # Calculate bounding box parameters
    min_coords = minimum(points, dims=2)
    max_coords = maximum(points, dims=2)
    box_lengths = max_coords .- min_coords
    longest_axis = argmax(box_lengths)[1]
    
    # Define cluster center and radius
    # center = (min_coords .+ max_coords) ./ 2
    center = sum(points, dims=2) / size(points, 2)
    radius = norm(center .- max_coords)
    radius = max(radius, norm(center .- min_coords))

    # Return if this is a leaf node
    if end_idx - start_idx <= max_points_per_leaf
        node = ClusterNode(true, start_idx, end_idx, vec(center), radius, nothing, nothing)
        return node
    end

    # Split along the longest axis for non-leaf nodes
    split_dimension_values = points[longest_axis, :]
    is_left_partition = split_dimension_values .< center[longest_axis]

    # Re-arrange indices for left and right partitions
    left_indices = idx_view[findall(is_left_partition)]
    right_indices = idx_view[findall(.!is_left_partition)]
    n_left = length(left_indices)

    idx_view[1:n_left] .= left_indices
    idx_view[n_left+1:end] .= right_indices

    # Create child nodes
    split_index = start_idx + n_left
    
    left = build_tree_node(coordinates, max_points_per_leaf, index_map, start_idx, split_index)
    right = build_tree_node(coordinates, max_points_per_leaf, index_map, split_index, end_idx)
    node = ClusterNode(false, start_idx, end_idx, vec(center), radius, left, right)
    return node
end

"""
    info(tree::ClusterTree) -> Dict{Symbol, Any}

Analyzes the given `ClusterTree` to extract basic statistics.

# Arguments
- `tree::ClusterTree`: The hierarchical clustering tree to be analyzed.

# Returns
A dictionary containing:
  - `depth`: Total number of depths in the tree.
  - `max_points`: Maximum number of points in any leaf node.
  - `min_points`: Minimum number of points in any leaf node.
"""
function info(tree::ClusterTree)
    # Initialize counters and storage for statistics
    depths = 0
    max_points = 0
    min_points = Inf  # Start with infinity to find the minimum
    
    # Recursive function to traverse the tree and gather stats
    function traverse(node::ClusterNode, current_depth::Int)
        
        # Update the depth and point counts
        depths = max(depths, current_depth)
        
        if node.is_leaf 
            points_count = node.end_idx - node.start_idx
            max_points = max(max_points, points_count)
            min_points = min(min_points, points_count)
        end
        
        # Traverse child nodes if they exist
        if node.left !== nothing
            traverse(node.left, current_depth + 1)
        end
        
        if node.right !== nothing
            traverse(node.right, current_depth + 1)
        end
    end
    
    # Start traversal from the root node at depth 1
    traverse(tree.root, 1)
    
    # Prepare the result as a dictionary
    return Dict(:depth => depths, :max_points => max_points, :min_points => min_points == Inf ? 0 : Int(min_points))
end

