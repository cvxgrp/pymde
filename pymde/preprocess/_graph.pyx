cimport numpy as cnp


ctypedef int ITYPE_t

# NULL_IDX is the index used in predecessor matrices to store a non-path
DEF NULL_IDX = -9999


def _breadth_first_directed(
                           unsigned int head_node,
                           cnp.ndarray[ITYPE_t, ndim=1, mode='c'] indices,
                           cnp.ndarray[ITYPE_t, ndim=1, mode='c'] indptr,
                           cnp.ndarray[ITYPE_t, ndim=1, mode='c'] node_list,
                           cnp.ndarray[ITYPE_t, ndim=1, mode='c'] lengths,
                           cnp.ndarray[ITYPE_t, ndim=1, mode='c'] predecessors):
    # Inputs:
    #  head_node: (input) index of the node from which traversal starts
    #  indices: (input) CSR indices of graph
    #  indptr:  (input) CSR indptr of graph
    #  node_list: (output) breadth-first list of nodes
    #  predecessors: (output) list of predecessors of nodes in breadth-first
    #                tree.  Should be initialized to NULL_IDX
    # Returns:
    #  n_nodes: the number of nodes in the breadth-first tree
    cdef unsigned int i, pnode, cnode
    cdef unsigned int i_nl, i_nl_end
    cdef unsigned int N = node_list.shape[0]
    cdef unsigned int curr_length

    node_list[0] = head_node
    lengths[head_node] = 0
    i_nl = 0
    i_nl_end = 1

    while i_nl < i_nl_end:
        pnode = node_list[i_nl]
        curr_length = lengths[pnode]

        for i in range(indptr[pnode], indptr[pnode + 1]):
            cnode = indices[i]
            if (cnode == head_node):
                continue
            elif (predecessors[cnode] == NULL_IDX):
                node_list[i_nl_end] = cnode
                predecessors[cnode] = pnode
                lengths[cnode] = curr_length + 1
                i_nl_end += 1

        i_nl += 1

    return i_nl


