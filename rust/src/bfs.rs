use numpy::PyReadonlyArray1;
use pyo3::prelude::*;

const NULL_IDX: i32 = -9999;

/// Breadth-first search on a directed CSR graph.
///
/// Mutates `node_list`, `lengths`, and `predecessors` in place.
/// Returns the number of nodes visited.
///
/// Parameters
/// ----------
/// head_node : int
///     Starting node index.
/// indices : numpy.ndarray[int32]
///     CSR column indices.
/// indptr : numpy.ndarray[int32]
///     CSR row pointers.
/// node_list : numpy.ndarray[int32]
///     Output: BFS traversal order.
/// lengths : numpy.ndarray[int32]
///     Output: BFS depth of each node.
/// predecessors : numpy.ndarray[int32]
///     Output: predecessor of each node in BFS tree.
///     Should be initialized to NULL_IDX (-9999).
#[pyfunction]
pub(crate) fn breadth_first_directed<'py>(
    _py: Python<'py>,
    head_node: u32,
    indices: PyReadonlyArray1<'py, i32>,
    indptr: PyReadonlyArray1<'py, i32>,
    mut node_list: numpy::PyReadwriteArray1<'py, i32>,
    mut lengths: numpy::PyReadwriteArray1<'py, i32>,
    mut predecessors: numpy::PyReadwriteArray1<'py, i32>,
) -> PyResult<usize> {
    let indices = indices.as_slice()?;
    let indptr = indptr.as_slice()?;
    let node_list = node_list.as_slice_mut()?;
    let lengths = lengths.as_slice_mut()?;
    let predecessors = predecessors.as_slice_mut()?;

    Ok(bfs_directed(
        head_node as usize,
        indices,
        indptr,
        node_list,
        lengths,
        predecessors,
    ))
}

/// Pure-Rust BFS on a CSR graph. Mirrors the Cython `_breadth_first_directed`.
fn bfs_directed(
    head_node: usize,
    indices: &[i32],
    indptr: &[i32],
    node_list: &mut [i32],
    lengths: &mut [i32],
    predecessors: &mut [i32],
) -> usize {
    node_list[0] = head_node as i32;
    lengths[head_node] = 0;

    let mut i_nl: usize = 0;
    let mut i_nl_end: usize = 1;

    while i_nl < i_nl_end {
        let pnode = node_list[i_nl] as usize;
        let curr_length = lengths[pnode];

        let start = indptr[pnode] as usize;
        let end = indptr[pnode + 1] as usize;
        for i in start..end {
            let cnode = indices[i] as usize;
            if cnode == head_node {
                continue;
            }
            if predecessors[cnode] == NULL_IDX {
                node_list[i_nl_end] = cnode as i32;
                predecessors[cnode] = pnode as i32;
                lengths[cnode] = curr_length + 1;
                i_nl_end += 1;
            }
        }

        i_nl += 1;
    }

    i_nl
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: run BFS and return (visited_count, node_list, lengths, predecessors).
    fn run_bfs(
        head: usize,
        indices: &[i32],
        indptr: &[i32],
        n: usize,
    ) -> (usize, Vec<i32>, Vec<i32>, Vec<i32>) {
        let mut node_list = vec![NULL_IDX; n];
        let mut lengths = vec![NULL_IDX; n];
        let mut predecessors = vec![NULL_IDX; n];
        let count = bfs_directed(
            head,
            indices,
            indptr,
            &mut node_list,
            &mut lengths,
            &mut predecessors,
        );
        (count, node_list, lengths, predecessors)
    }

    #[test]
    fn bfs_linear_graph() {
        // 0 -> 1 -> 2 -> 3
        let indptr = vec![0, 1, 2, 3, 3]; // node 3 has no outgoing edges
        let indices = vec![1, 2, 3];
        let (count, node_list, lengths, predecessors) = run_bfs(0, &indices, &indptr, 4);
        assert_eq!(count, 4);
        assert_eq!(&node_list[..4], &[0, 1, 2, 3]);
        assert_eq!(lengths[0], 0);
        assert_eq!(lengths[1], 1);
        assert_eq!(lengths[2], 2);
        assert_eq!(lengths[3], 3);
        assert_eq!(predecessors[0], NULL_IDX); // head has no predecessor
        assert_eq!(predecessors[1], 0);
        assert_eq!(predecessors[2], 1);
        assert_eq!(predecessors[3], 2);
    }

    #[test]
    fn bfs_star_graph() {
        // 0 -> 1, 0 -> 2, 0 -> 3
        let indptr = vec![0, 3, 3, 3, 3];
        let indices = vec![1, 2, 3];
        let (count, _node_list, lengths, predecessors) = run_bfs(0, &indices, &indptr, 4);
        assert_eq!(count, 4);
        assert_eq!(lengths[0], 0);
        for i in 1..4 {
            assert_eq!(lengths[i], 1);
            assert_eq!(predecessors[i], 0);
        }
    }

    #[test]
    fn bfs_disconnected() {
        // 0 -> 1, nodes 2 and 3 unreachable
        let indptr = vec![0, 1, 1, 2, 2];
        let indices = vec![1, 3]; // edge 2->3
        let (count, node_list, lengths, predecessors) = run_bfs(0, &indices, &indptr, 4);
        assert_eq!(count, 2); // only 0 and 1 reachable
        assert_eq!(node_list[0], 0);
        assert_eq!(node_list[1], 1);
        assert_eq!(lengths[2], NULL_IDX);
        assert_eq!(lengths[3], NULL_IDX);
        assert_eq!(predecessors[2], NULL_IDX);
    }

    #[test]
    fn bfs_cycle() {
        // Undirected cycle: 0-1-2-3-0 (as symmetric CSR)
        // indptr: [0, 2, 4, 6, 8]
        // indices: [1,3, 0,2, 1,3, 2,0]
        let indptr = vec![0, 2, 4, 6, 8];
        let indices = vec![1, 3, 0, 2, 1, 3, 2, 0];
        let (count, _node_list, lengths, _preds) = run_bfs(0, &indices, &indptr, 4);
        assert_eq!(count, 4);
        assert_eq!(lengths[0], 0);
        assert_eq!(lengths[1], 1);
        assert_eq!(lengths[3], 1);
        assert_eq!(lengths[2], 2);
    }

    #[test]
    fn bfs_single_node() {
        let indptr = vec![0, 0];
        let indices: Vec<i32> = vec![];
        let (count, node_list, lengths, _) = run_bfs(0, &indices, &indptr, 1);
        assert_eq!(count, 1);
        assert_eq!(node_list[0], 0);
        assert_eq!(lengths[0], 0);
    }
}
