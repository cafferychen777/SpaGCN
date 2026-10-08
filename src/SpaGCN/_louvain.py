"""Pure-Python Louvain clustering that reproduces ``scanpy.tl.louvain``.

SpaGCN 1.2.7 initialized its clusters with ``scanpy.tl.louvain(adata,
resolution=res)``, which delegates to the ``louvain`` package (V. A. Traag,
0.8.x). That package is no longer maintained, ships no wheels for current
Python releases, and pins ``igraph<0.12``, so it cannot be installed next to
current Scanpy/leidenalg stacks.

This module is an independent implementation of the Louvain method (Blondel et
al., 2008) for the exact configuration Scanpy uses by default: the unweighted,
directed kNN connectivity graph, the Reichardt-Bornholdt configuration-model
quality function with a linear resolution parameter, all neighbouring
communities considered for every move, moves to an empty community allowed,
and random seed 0. It follows the same visiting order (Fisher-Yates shuffle
driven by a 32-bit Mersenne Twister with Lemire's bounded integers), the same
tie-breaking and the same community numbering, so it returns the same
membership vector as ``louvain.find_partition``.

Edge weights are unit counts at the first level and integer sums after
aggregation, so every weight total is exact in floating point and only the
quality differences themselves involve rounding.
"""

import numpy as np
from scipy import sparse

_MT_N = 624
_MT_M = 397
_UINT32 = 0xFFFFFFFF


class _MersenneTwister:
    """MT19937 with Knuth's initialization, as used by igraph's RNG."""

    def __init__(self, seed):
        seed &= _UINT32
        if seed == 0:
            seed = 4357
        state = [seed]
        for i in range(1, _MT_N):
            prev = state[-1]
            state.append((1812433253 * (prev ^ (prev >> 30)) + i) & _UINT32)
        self._state = state
        self._index = _MT_N

    def _twist(self):
        mt = self._state
        for k in range(_MT_N):
            y = (mt[k] & 0x80000000) | (mt[(k + 1) % _MT_N] & 0x7FFFFFFF)
            value = mt[(k + _MT_M) % _MT_N] ^ (y >> 1)
            if y & 1:
                value ^= 0x9908B0DF
            mt[k] = value
        self._index = 0

    def next_uint32(self):
        if self._index >= _MT_N:
            self._twist()
        y = self._state[self._index]
        self._index += 1
        y ^= y >> 11
        y ^= (y << 7) & 0x9D2C5680
        y ^= (y << 15) & 0xEFC60000
        y ^= y >> 18
        return y

    def integer(self, low, high):
        """Uniform integer in ``[low, high]`` (Lemire's debiased method)."""
        if high == low:
            return low
        span = high - low + 1
        threshold = ((1 << 32) - span) % span
        while True:
            product = self.next_uint32() * span
            if (product & _UINT32) >= threshold:
                return low + (product >> 32)

    def shuffle(self, values):
        for idx in range(len(values) - 1, 0, -1):
            swap = self.integer(0, idx)
            values[idx], values[swap] = values[swap], values[idx]


class _Graph:
    """Directed multigraph level with integer edge weights."""

    def __init__(self, n, node_sizes, out_edges):
        # out_edges[v] is a list of (target, weight) pairs.
        self.n = n
        self.node_sizes = node_sizes
        self.out_edges = out_edges
        in_edges = [[] for _ in range(n)]
        self_weights = [0] * n
        strength_out = [0] * n
        strength_in = [0] * n
        total = 0
        for v in range(n):
            for u, w in out_edges[v]:
                in_edges[u].append((v, w))
                strength_out[v] += w
                strength_in[u] += w
                total += w
                if u == v:
                    self_weights[v] = w
        self.in_edges = in_edges
        self.self_weights = self_weights
        self.strength_out = strength_out
        self.strength_in = strength_in
        self.total_weight = total

    @classmethod
    def from_adjacency(cls, adjacency):
        adjacency = sparse.csr_matrix(adjacency)
        sources, targets = adjacency.nonzero()
        n = adjacency.shape[0]
        out_edges = [[] for _ in range(n)]
        for v, u in zip(sources.tolist(), targets.tolist()):
            out_edges[v].append((u, 1))
        return cls(n, [1] * n, out_edges)

    def collapse(self, partition):
        """Aggregate each community into one node (weights are summed)."""
        membership = partition.membership
        n_comms = partition.n_communities
        weights = [dict() for _ in range(n_comms)]
        for v in range(self.n):
            v_comm = membership[v]
            row = weights[v_comm]
            for u, w in self.out_edges[v]:
                u_comm = membership[u]
                row[u_comm] = row.get(u_comm, 0) + w
        out_edges = [sorted(row.items()) for row in weights]
        return _Graph(n_comms, list(partition.community_sizes), out_edges)


class _Partition:
    """Reichardt-Bornholdt configuration partition with a resolution."""

    def __init__(self, graph, resolution, membership=None):
        self.graph = graph
        self.resolution = resolution
        if membership is None:
            membership = list(range(graph.n))
        self.membership = list(membership)
        self.empty_communities = []
        self._init_admin()

    @property
    def n_communities(self):
        return len(self.node_counts)

    def _init_admin(self):
        graph = self.graph
        n_comms = max(self.membership) + 1 if self.membership else 0
        self.node_counts = [0] * n_comms
        self.community_sizes = [0] * n_comms
        self.weight_from_comm = [0] * n_comms
        self.weight_to_comm = [0] * n_comms
        for v, comm in enumerate(self.membership):
            self.node_counts[comm] += 1
            self.community_sizes[comm] += graph.node_sizes[v]
            self.weight_from_comm[comm] += graph.strength_out[v]
            self.weight_to_comm[comm] += graph.strength_in[v]
        # The reference implementation keeps previously recorded empty
        # communities when the administration is rebuilt.
        for comm in range(n_comms):
            if self.node_counts[comm] == 0:
                self.empty_communities.append(comm)
        self._cache_out = (None, None)
        self._cache_in = (None, None)
        self._cache_all = (None, None)

    def _community_weights(self, v, edges):
        weights = {}
        membership = self.membership
        for u, w in edges:
            comm = membership[u]
            weights[comm] = weights.get(comm, 0) + w
        return weights

    def neighbour_communities(self, v):
        node, comms = self._cache_all
        if node != v:
            membership = self.membership
            comms = {membership[u] for u, _ in self.graph.out_edges[v]}
            comms.update(membership[u] for u, _ in self.graph.in_edges[v])
            comms = sorted(comms)
            self._cache_all = (v, comms)
        return comms

    def _weight_to(self, v, comm):
        node, weights = self._cache_out
        if node != v:
            weights = self._community_weights(v, self.graph.out_edges[v])
            self._cache_out = (v, weights)
        return weights.get(comm, 0)

    def _weight_from(self, v, comm):
        node, weights = self._cache_in
        if node != v:
            weights = self._community_weights(v, self.graph.in_edges[v])
            self._cache_in = (v, weights)
        return weights.get(comm, 0)

    def diff_move(self, v, new_comm):
        graph = self.graph
        total_weight = graph.total_weight
        if total_weight == 0:
            return 0.0
        old_comm = self.membership[v]
        if new_comm == old_comm:
            return 0.0
        res = self.resolution
        w_to_old = self._weight_to(v, old_comm)
        w_from_old = self._weight_from(v, old_comm)
        w_to_new = self._weight_to(v, new_comm)
        w_from_new = self._weight_from(v, new_comm)
        k_out = graph.strength_out[v]
        k_in = graph.strength_in[v]
        self_weight = graph.self_weights[v]
        k_out_old = self.weight_from_comm[old_comm]
        k_in_old = self.weight_to_comm[old_comm]
        k_out_new = self.weight_from_comm[new_comm] + k_out
        k_in_new = self.weight_to_comm[new_comm] + k_in
        diff_old = (w_to_old - res * k_out * k_in_old / total_weight) + (
            w_from_old - res * k_in * k_out_old / total_weight
        )
        diff_new = (w_to_new + self_weight - res * k_out * k_in_new / total_weight) + (
            w_from_new + self_weight - res * k_in * k_out_new / total_weight
        )
        return diff_new - diff_old

    def _add_empty_community(self):
        if self.n_communities + 1 > self.graph.n:
            raise RuntimeError("cannot create more communities than nodes")
        self.node_counts.append(0)
        self.community_sizes.append(0)
        self.weight_from_comm.append(0)
        self.weight_to_comm.append(0)
        comm = self.n_communities - 1
        self.empty_communities.append(comm)
        return comm

    def empty_community(self):
        if not self.empty_communities:
            self._add_empty_community()
        return self.empty_communities[-1]

    def move_node(self, v, new_comm):
        while new_comm >= self.n_communities:
            self._add_empty_community()
        graph = self.graph
        old_comm = self.membership[v]
        size = graph.node_sizes[v]
        self.node_counts[old_comm] -= 1
        self.community_sizes[old_comm] -= size
        if self.node_counts[old_comm] == 0:
            self.empty_communities.append(old_comm)
        if self.node_counts[new_comm] == 0:
            empty = self.empty_communities
            for idx in range(len(empty) - 1, -1, -1):
                if empty[idx] == new_comm:
                    del empty[idx]
                    break
        self.node_counts[new_comm] += 1
        self.community_sizes[new_comm] += size
        self.weight_from_comm[old_comm] -= graph.strength_out[v]
        self.weight_from_comm[new_comm] += graph.strength_out[v]
        self.weight_to_comm[old_comm] -= graph.strength_in[v]
        self.weight_to_comm[new_comm] += graph.strength_in[v]
        self.membership[v] = new_comm

    def renumbered_membership(self):
        """Largest community first; ties by node count, then old label."""
        order = sorted(
            range(self.n_communities),
            key=lambda c: (-self.community_sizes[c], -self.node_counts[c], c),
        )
        new_ids = [0] * self.n_communities
        for rank, comm in enumerate(order):
            new_ids[comm] = rank
        return [new_ids[comm] for comm in self.membership]

    def set_membership(self, membership):
        self.membership = list(membership)
        self._init_admin()


def _move_nodes(partition, rng, eps=1e-10):
    nodes = list(range(partition.graph.n))
    rng.shuffle(nodes)
    total_improvement = 0.0
    n_moves = 1
    while n_moves > 0:
        improvement = 0.0
        n_moves = 0
        for v in nodes:
            v_comm = partition.membership[v]
            best_comm = v_comm
            best_improvement = 0.0
            for comm in partition.neighbour_communities(v):
                candidate = partition.diff_move(v, comm)
                if candidate > best_improvement and candidate > eps:
                    best_comm = comm
                    best_improvement = candidate
            if partition.node_counts[v_comm] > 1:
                comm = partition.empty_community()
                candidate = partition.diff_move(v, comm)
                if candidate > best_improvement:
                    best_improvement = candidate
                    best_comm = comm
            if best_comm != v_comm:
                improvement += best_improvement
                partition.move_node(v, best_comm)
                n_moves += 1
        total_improvement += improvement
    partition.set_membership(partition.renumbered_membership())
    return total_improvement


def louvain_membership(adjacency, resolution=1.0, seed=0):
    """Louvain communities of the unweighted, directed graph ``adjacency``.

    Equivalent to ``louvain.find_partition(g, RBConfigurationVertexPartition,
    resolution_parameter=resolution, seed=seed)`` on the graph that
    ``scanpy.tl.louvain`` builds from ``adjacency`` by default.
    """
    rng = _MersenneTwister(int(seed))
    graph = _Graph.from_adjacency(adjacency)
    partition = _Partition(graph, resolution)
    level_graph = graph
    level_partition = partition
    while True:
        improvement = _move_nodes(level_partition, rng)
        if level_partition is not partition:
            coarse = level_partition.membership
            partition.set_membership([coarse[c] for c in partition.membership])
        level_graph = level_graph.collapse(level_partition)
        level_partition = _Partition(level_graph, resolution)
        if not improvement > 0:
            break
    return np.asarray(partition.renumbered_membership(), dtype=np.int64)
