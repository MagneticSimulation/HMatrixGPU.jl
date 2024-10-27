# Structure defining a node within the block tree, representing a block in the hierarchical matrix
mutable struct BlockTreeNode
    D::Union{Nothing, Matrix{Float64}}      # Dense matrix if this node is dense
    U::Union{Nothing, Matrix{Float64}}      # U matrix for low-rank approximation (RkMatrix)
    V::Union{Nothing, Matrix{Float64}}      # V matrix for low-rank approximation (RkMatrix)
    target_tree::Union{Nothing, ClusterNode}  # Target cluster node (from target tree)
    source_tree::Union{Nothing, ClusterNode}  # Source cluster node (from source tree)
    is_admissible::Bool                     # Indicates if node is admissible (approximable by low-rank)
    is_dense::Bool                          # Indicates if node represents a dense matrix
    left::Union{Nothing, BlockTreeNode}     # Left child node for recursive block subdivision
    right::Union{Nothing, BlockTreeNode}    # Right child node for recursive block subdivision
end

# Structure representing the entire block tree for hierarchical matrices
struct BlockTree
    K::AbstractMatrix                       # Matrix for which the block tree is being constructed
    target_index_map::Vector{Int}           # Mapping of target indices for reordering
    source_index_map::Vector{Int}           # Mapping of source indices for reordering
    root::BlockTreeNode                     # Root node of the block tree
end

"""
    BlockTree(K::AbstractMatrix, X::ClusterTree, Y::ClusterTree; eta=1.5)

Constructs a block tree for the given matrix `K` based on the target and source cluster trees `X` and `Y`.
The parameter `eta` controls the admissibility condition, defining when a node is considered low-rank
and can be represented by a low-rank approximation.

# Arguments
- `K`: Matrix to decompose.
- `X`: Cluster tree representing the target points.
- `Y`: Cluster tree representing the source points.
- `eta`: Threshold parameter controlling the admissibility condition.

# Returns
 A `BlockTree` structure, with the root node representing the hierarchical decomposition of `K`.
"""
function BlockTree(K::AbstractMatrix, X::ClusterTree, Y::ClusterTree; eta=1.5)
    # Initialize the root node with the target and source cluster roots
    root = BlockTreeNode(nothing, nothing, nothing, X.root, Y.root, false, false, nothing, nothing)
    
    # Recursively build the block tree
    build_block_tree!(root, eta)

    # Return the constructed BlockTree with target and source index mappings
    return BlockTree(K, X.index_map, Y.index_map, root)
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
            node.left = BlockTreeNode(nothing, nothing, nothing, X, Y.left, false, false, nothing, nothing)
            node.right = BlockTreeNode(nothing, nothing, nothing, X, Y.right, false, false, nothing, nothing)
        else
            node.left = BlockTreeNode(nothing, nothing, nothing, X.left, Y, false, false, nothing, nothing)
            node.right = BlockTreeNode(nothing, nothing, nothing, X.right, Y, false, false, nothing, nothing)
        end
        
        # Recursively build the left and right subtrees
        build_block_tree!(node.left, eta)
        build_block_tree!(node.right, eta)
    end
end
