# Structure defining a node within the block tree, representing a block in the hierarchical matrix
mutable struct BlockTreeNode
    D::Union{Nothing,Matrix{Float64}}      # Dense matrix if this node is dense
    U::Union{Nothing,Matrix{Float64}}      # U matrix for low-rank approximation (RkMatrix)
    V::Union{Nothing,Matrix{Float64}}      # V matrix for low-rank approximation (RkMatrix)
    target_tree::Union{Nothing,ClusterNode}  # Target cluster node (from target tree)
    source_tree::Union{Nothing,ClusterNode}  # Source cluster node (from source tree)
    is_admissible::Bool                     # Indicates if node is admissible (approximable by low-rank)
    is_dense::Bool                          # Indicates if node represents a dense matrix
    left::Union{Nothing,BlockTreeNode}     # Left child node for recursive block subdivision
    right::Union{Nothing,BlockTreeNode}    # Right child node for recursive block subdivision
end

# Structure representing the entire block tree for hierarchical matrices
struct BlockTree
    target_index_map::Vector{Int}           # Mapping of target indices for reordering
    source_index_map::Vector{Int}           # Mapping of source indices for reordering
    root::BlockTreeNode                     # Root node of the block tree
end

"""
    BlockTree(X::ClusterTree, Y::ClusterTree; eta=1.5)

Constructs a block tree based on the target and source cluster trees `X` and `Y`.
The parameter `eta` controls the admissibility condition, defining when a node is considered low-rank
and can be represented by a low-rank approximation.

# Arguments
- `X`: Cluster tree representing the target points.
- `Y`: Cluster tree representing the source points.
- `eta`: Threshold parameter controlling the admissibility condition.

"""
function BlockTree(X::ClusterTree, Y::ClusterTree; eta=1.5)
    # Initialize the root node with the target and source cluster roots
    root = BlockTreeNode(nothing, nothing, nothing, X.root, Y.root, false, false, nothing,
                         nothing)

    # Recursively build the block tree
    build_block_tree!(root, eta)

    # Return the constructed BlockTree with target and source index mappings
    return BlockTree(X.index_map, Y.index_map, root)
end

"""
    build_block_tree!(node::BlockTreeNode, eta)

Recursively constructs the block tree by subdividing nodes based on the admissibility condition.
If the distance between target and source clusters meets the admissibility condition, the node
is flagged as admissible and can use a low-rank approximation. Otherwise, it is subdivided further
until a leaf node is reached.

# Arguments
- `node`: Current `BlockTreeNode` being evaluated.
- `eta`: Threshold parameter for admissibility.
"""
function build_block_tree!(node::BlockTreeNode, eta)
    X = node.target_tree
    Y = node.source_tree

    # Compute distance between the centers of the target and source clusters
    dist = norm(X.center .- Y.center)

    # Admissibility condition check: if true, mark node as admissible
    if dist > eta * (X.radius + Y.radius)
        node.is_admissible = true
        return
        # If both clusters are leaves, mark the node as dense
    elseif X.is_leaf && Y.is_leaf
        node.is_dense = true
        return
    else
        # Determine whether to split the target or source cluster based on size
        split_Y = (!Y.is_leaf && Y.radius > X.radius) || X.is_leaf

        # Create child nodes by splitting target or source cluster as determined
        if split_Y
            node.left = BlockTreeNode(nothing, nothing, nothing, X, Y.left, false, false,
                                      nothing, nothing)
            node.right = BlockTreeNode(nothing, nothing, nothing, X, Y.right, false, false,
                                       nothing, nothing)
        else
            node.left = BlockTreeNode(nothing, nothing, nothing, X.left, Y, false, false,
                                      nothing, nothing)
            node.right = BlockTreeNode(nothing, nothing, nothing, X.right, Y, false, false,
                                       nothing, nothing)
        end

        # Recursively build the left and right subtrees
        build_block_tree!(node.left, eta)
        build_block_tree!(node.right, eta)
    end
end

"""
    merge_dense_matrices!(node::Union{Nothing, BlockTreeNode})

Recursively merges dense matrix flags for a `BlockTreeNode`. If a node's children are dense, 
it updates the node's `is_dense` flag accordingly.
"""
function merge_dense_matrices!(node::Union{Nothing,BlockTreeNode})
    # Return if node is empty
    if node === nothing
        return
    end

    # Recursively process child nodes
    if node.left !== nothing && node.right !== nothing
        merge_dense_matrices!(node.left)
        merge_dense_matrices!(node.right)

        # Set `is_dense` to true if both children are dense
        node.is_dense = node.left.is_dense && node.right.is_dense
    end
end

"""
    traverse(block_tree::BlockTree)

Traverses a `BlockTree` to categorize nodes as either `direct_list` for dense (direct) 
interactions or `approx_list` for admissible (approximate) interactions.

# Arguments
- `block_tree::BlockTree`: The block tree to traverse.

# Returns
- `Tuple{Vector, Vector}`: Two vectors, `direct_list` and `approx_list`, containing tuples 
  of target and source trees.
"""
function traverse(block_tree::BlockTree)
    # Initialize lists for categorizing dense and admissible nodes
    direct_list, approx_list = Vector{Tuple{ClusterNode,ClusterNode}}(),
                               Vector{Tuple{ClusterNode,ClusterNode}}()

    # Recursive helper function to categorize nodes within the block tree based on density and admissibility.
    function _traverse(node::BlockTreeNode, direct_list, approx_list)
        # Categorize nodes as dense (direct interaction) or admissible (approximate interaction)
        if node.is_admissible
            push!(approx_list, (node.target_tree, node.source_tree))
        elseif node.is_dense
            push!(direct_list, (node.target_tree, node.source_tree))
        end

        # Recurse to left and right children if node is neither dense nor admissible
        if !node.is_dense && !node.is_admissible
            if node.left !== nothing
                _traverse(node.left, direct_list, approx_list)
            end
            if node.right !== nothing
                _traverse(node.right, direct_list, approx_list)
            end
        end
    end

    # Start the recursive traversal from the root of the block tree
    _traverse(block_tree.root, direct_list, approx_list)
    return direct_list, approx_list
end
