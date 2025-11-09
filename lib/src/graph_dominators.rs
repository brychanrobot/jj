// Copyright 2026 The Jujutsu Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Generic implementation of the "closest common dominator" algorithm for
//! directed graphs.

use std::collections::HashMap;
use std::collections::HashSet;
use std::collections::VecDeque;
use std::hash::Hash;

use indexmap::IndexMap;
use indexmap::IndexSet;
use itertools::Itertools as _;
use thiserror::Error;

/// Generic implementation of the Common Dominator algorithm for directed
/// graphs, using the Cooper-Harvey-Kennedy iterative algorithm. Loosely
/// speaking the algorithm finds the "choke point" for a set of nodes S in a
/// directed graph (going from the "entry" node to nodes in S), closest to S.
///
/// Dominance:
///
/// * An entry node is a node with no incoming edges.
/// * A graph may have zero or more entry nodes. In any case, a virtual node can
///   be added to a graph to make it have a unique entry node.
/// * A node z is said to dominate a node n if all paths from the unique entry
///   node to n must go through z. Every node dominates itself, and the (unique)
///   entry node dominates all nodes.
/// * A node can have one or more dominators.
/// * A node z strictly dominates n if z dominates n and z != n.
/// * The immediate dominator of a node n is the dominator of n that doesn't
///   strictly dominate any other strict dominators of n. Informally it is the
///   "closest" choke point on all paths from the entry node to n.
/// * Let S be a subset of the nodes in the graph. The intersection of the
///   dominators of each node in S is the set of common dominators of S.
/// * The closest common dominator of S is the common dominator of S that
///   doesn't strictly dominate any other common dominator of S. Informally, it
///   is the choke point closest to S such that all paths from the entry node to
///   S must go through it.
///
/// Dominator Tree:
///
/// For any directed graph G with a single entry node there is a corresponding
/// dominator tree defined as follows:
/// * The nodes of the dominator tree are the same as the nodes of G
/// * The root of the dominator tree is the entry node of G
/// * In the dominator tree, the children of a node are the nodes it immediately
///   dominates
///
/// The closest common dominator of S is the Lowest Common Ancestor (LCA)
/// of S in the graph's dominator tree.
///
/// This implementation constructs the Dominator Tree by first determining
/// the Immediate Dominator (ipdom) for every node (using the standard iterative
/// algorithm), and then calculating the LCA for the set S. See:
///
/// * <http://www.hipersoft.rice.edu/grads/publications/dom14.pdf>
/// * <https://en.wikipedia.org/wiki/Dominator_(graph_theory)>
///
/// The running time is O(V+E+|S|*V)in the worst case, the space complexity is
/// O(V+E), where V is the number of nodes and E is the number of edges.
///
/// For a DAG, the expensive iterative Dominator step converges in just two
/// passes, making the tree construction effectively linear, O(V+E). The primary
/// bottleneck becomes the naive "Lowest Common Ancestor" (LCA) lookup, which
/// scales with the size of the set and the depth of the tree. If you need
/// better running time for large graphs, you can optimize the LCA step.
///
/// T must be Hash + Eq to be used as a key, and Clone to be returned.
struct DominatorFinder<T> {
    // Nodes are given consecutive integer IDs internally for efficient graph algorithms.
    node_to_id: HashMap<T, Index>,
    // Maps internal IDs back to the original generic type T for output.
    id_to_node: Vec<T>,
    // Forward adjacency list: adj[u] = [v1, v2, ...] means there are edges u->v1, u->v2, ...
    // Includes a virtual entry node that points to all natural entry nodes (in the original
    // graph), ensuring a single entry for the graph.
    adj: Vec<Vec<Index>>,
    // Reverse adjacency list.
    // Includes the virtual entry node.
    rev_adj: Vec<Vec<Index>>,
}

/// Type alias for clarity.
type Index = usize;

/// Errors that can occur during dominator finding.
#[derive(Debug, Error, PartialEq)]
pub enum DominatorFinderError {
    /// Edge contains unknown node.
    #[error("Invalid input: edge contains unknown node")]
    EdgeContainsUnknownNode,
    /// Target set contains unknown node.
    #[error("Invalid input: target set contains unknown node")]
    TargetSetContainsUnknownNode,
    /// Graph has no entry node.
    #[error("Graph has no entry node")]
    GraphHasNoEntryNode,
}

/// Finds the closest common dominator for the given graph and set of nodes S
/// (target_set).
pub fn find_closest_common_dominator<T, NI, EI, TI>(
    nodes: NI,
    edges: EI,
    target_set: TI,
) -> Result<Option<T>, DominatorFinderError>
where
    T: Hash + Eq + Clone,
    NI: IntoIterator<Item = T>,
    EI: IntoIterator<Item = (T, T)>,
    TI: IntoIterator<Item = T>,
{
    DominatorFinder::new(nodes, edges)?.closest_common_dominator(target_set)
}

impl<T> DominatorFinder<T>
where
    T: Hash + Eq + Clone,
{
    /// Constructs a new DominatorFinder from a list of nodes and edges.
    /// The edges must correspond to the nodes provided; (u, v) means u->v.
    ///
    /// Returns an error if edges contain unknown nodes, or if the graph does
    /// not have any entry nodes (i.e., every node has at least one incoming
    /// edge). In the latter case, clients can add a virtual entry node
    /// themselves before calling this constructor. Clearly this is never an
    /// issue for DAGs.
    fn new<NI, EI>(nodes: NI, edges: EI) -> Result<Self, DominatorFinderError>
    where
        NI: IntoIterator<Item = T>,
        EI: IntoIterator<Item = (T, T)>,
    {
        let mut node_to_id = HashMap::new();
        let mut id_to_node = Vec::new();

        // 1. Map generic types to integer IDs
        let mut i = 0;
        for node in nodes {
            match node_to_id.entry(node.clone()) {
                std::collections::hash_map::Entry::Vacant(entry) => {
                    entry.insert(i);
                    id_to_node.push(node);
                    i += 1;
                }
                _ => {
                    // Skip duplicates.
                }
            }
        }
        let n_original = i; // The number of unique nodes (excluding the virtual entry).
        let virtual_entry = n_original;

        let mut adj = vec![vec![]; n_original + 1]; // Reserve space for virtual entry
        let mut rev_adj = vec![vec![]; n_original + 1]; // Reserve space for virtual entry
        let mut seen_edges = HashSet::new();

        // 2. Build Graph using internal IDs
        for (u, v) in edges {
            if let (Some(&u), Some(&v)) = (node_to_id.get(&u), node_to_id.get(&v)) {
                if u == v {
                    continue; // Ignore self loops, they don't affect dominance.
                }
                if seen_edges.insert((u, v)) {
                    adj[u].push(v);
                    rev_adj[v].push(u);
                }
            } else {
                return Err(DominatorFinderError::EdgeContainsUnknownNode);
            }
        }

        // 3: Augment Graph with a unique virtual entry node. Note that we do this even
        // if the input graph already has a single entry.

        // Connect sources to the Virtual Entry
        let mut has_entry_node = false;
        for (i, predecessors) in rev_adj.iter_mut().enumerate().take(n_original) {
            if predecessors.is_empty() {
                predecessors.push(virtual_entry);
                adj[virtual_entry].push(i);
                has_entry_node = true;
            }
        }

        if !has_entry_node {
            return Err(DominatorFinderError::GraphHasNoEntryNode);
        }

        Ok(Self {
            node_to_id,
            id_to_node,
            adj,
            rev_adj,
        })
    }

    /// Finds the closest common dominator for the given set of nodes S
    /// (target_set). Returns None if the closest common dominator in the
    /// augmented graph is the virtual entry. Returns an error if any node in
    /// target_set is unknown.
    fn closest_common_dominator<TI>(
        &self,
        target_set: TI,
    ) -> Result<Option<T>, DominatorFinderError>
    where
        TI: IntoIterator<Item = T>,
    {
        // Convert generic inputs to internal IDs
        let target_set: Vec<Index> = target_set
            .into_iter()
            .map(|node| match self.node_to_id.get(&node) {
                Some(&id) => Ok(id),
                None => Err(DominatorFinderError::TargetSetContainsUnknownNode),
            })
            .try_collect()?;

        if target_set.is_empty() {
            return Ok(None);
        }

        // Step 1: Compute Dominators on Reverse Graph
        let n_original = self.adj.len() - 1;
        let virtual_entry = n_original; // The last node is the virtual entry
        let execution_order = Self::get_reverse_post_order(&self.adj, virtual_entry);

        // idom is the immediate dominator for each node.
        let mut idom: Vec<Option<Index>> = vec![None; n_original + 1];
        idom[virtual_entry] = Some(virtual_entry);

        loop {
            let mut changed = false;
            for &u in &execution_order {
                if u == virtual_entry {
                    continue;
                }

                // Process predecessors (nodes that flow INTO u).
                let preds = &self.rev_adj[u];
                if preds.is_empty() {
                    continue;
                }

                let mut new_idom = None;
                // Find first processed predecessor.
                for &p in preds {
                    if idom[p].is_some() {
                        new_idom = Some(p);
                        break;
                    }
                }

                if let Some(mut candidate) = new_idom {
                    for &p in preds {
                        if idom[p].is_some() {
                            candidate = Self::intersect(candidate, p, &idom, &execution_order);
                        }
                    }

                    if idom[u] != Some(candidate) {
                        idom[u] = Some(candidate);
                        changed = true;
                    }
                }
            }

            if !changed {
                break;
            }
        }

        // Step 3: Find LCA in Dominator Tree
        let mut current_lca = target_set[0];
        for &node in target_set.iter().skip(1) {
            current_lca = Self::find_lca(current_lca, node, &idom, virtual_entry);
        }

        if current_lca == virtual_entry {
            return Ok(None); // Only common dominator is the artificial root
        }

        // Map internal ID back to generic type T
        Ok(Some(self.id_to_node[current_lca].clone()))
    }

    fn intersect(mut b1: Index, mut b2: Index, idom: &[Option<Index>], order: &[Index]) -> Index {
        let get_order_idx =
            |node: Index| -> Index { order.iter().position(|&x| x == node).unwrap() };

        while b1 != b2 {
            while get_order_idx(b1) > get_order_idx(b2) {
                b1 = idom[b1].unwrap();
            }
            while get_order_idx(b2) > get_order_idx(b1) {
                b2 = idom[b2].unwrap();
            }
        }
        b1
    }

    fn find_lca(u: Index, v: Index, idom: &[Option<Index>], root: Index) -> Index {
        let mut path_u = HashSet::new();
        let mut curr = u;
        loop {
            path_u.insert(curr);
            if curr == root {
                break;
            }
            match idom[curr] {
                Some(p) if p != curr => curr = p,
                _ => break,
            }
        }

        let mut curr = v;
        loop {
            if path_u.contains(&curr) {
                return curr;
            }
            if curr == root {
                break;
            }
            match idom[curr] {
                Some(p) if p != curr => curr = p,
                _ => break,
            }
        }
        root
    }

    fn get_reverse_post_order(graph: &[Vec<Index>], root: Index) -> Vec<Index> {
        let mut visited = HashSet::new();
        let mut order = Vec::new();
        Self::dfs(root, graph, &mut visited, &mut order);
        order.reverse();
        order
    }

    fn dfs(u: Index, graph: &[Vec<Index>], visited: &mut HashSet<Index>, order: &mut Vec<Index>) {
        visited.insert(u);
        for &v in &graph[u] {
            if !visited.contains(&v) {
                Self::dfs(v, graph, visited, order);
            }
        }
        order.push(u);
    }
}

/// An immutable directed graph with nodes of type N and a minimal interface for
/// iterating over nodes and their adjacent nodes.
/// Note: multi-edges cannot be represented by this data structure.
#[derive(Clone, Eq, PartialEq, Debug)]
pub struct SimpleDirectedGraph<N>
where
    N: Clone + Eq + Hash + PartialEq,
{
    /// The adjacency map of the graph. Each key is a node, and the
    /// corresponding value is the set of adjacent nodes (i.e., the children of
    /// the key node). The adjacency map is in canonical form: for every
    /// u->v edge, there is an entry in adj with key v (even if v has no
    /// outgoing edges).
    adj: IndexMap<N, IndexSet<N>>,
}

impl<N> SimpleDirectedGraph<N>
where
    N: Clone + Eq + Hash + PartialEq,
{
    /// Constructs a new SimpleDirectedGraph from an adjacency map.
    /// Note: if necessary, the input map is canonicalized, preserving iteration
    /// order.
    pub fn new(mut adj: IndexMap<N, IndexSet<N>>) -> Self {
        let mut missing_nodes = IndexSet::new();
        for (_, children) in &adj {
            for child in children {
                if !adj.contains_key(child) {
                    missing_nodes.insert(child.clone());
                }
            }
        }
        for node in missing_nodes {
            adj.entry(node).or_default();
        }
        Self { adj }
    }

    /// Returns the nodes in this graph.
    pub fn nodes(&self) -> impl Iterator<Item = &N> {
        self.adj.keys()
    }

    /// Returns the edges in this graph.
    pub fn edges(&self) -> impl Iterator<Item = (&N, &N)> {
        self.adj
            .iter()
            .flat_map(|(parent, adj_set)| adj_set.iter().map(move |child| (parent, child)))
    }

    /// Returns the adjacent nodes for the given node, or None if the node is
    /// not in the graph.
    pub fn adjacent_nodes(&self, node: &N) -> Option<impl Iterator<Item = &N>> {
        self.adj.get(node).map(|adj_set| adj_set.iter())
    }

    /// Returns true if this graph contains the given node.
    pub fn contains_node(&self, node: &N) -> bool {
        self.adj.contains_key(node)
    }

    /// Constructs a new graph from a list of edges. Iteration order is
    /// preserved from the input. Multi-edges in the input are removed.
    pub fn from_edge_list<EI>(edges: EI) -> Self
    where
        EI: IntoIterator<Item = (N, N)>,
    {
        let mut nodes: IndexSet<N> = IndexSet::new();
        let mut adj: IndexMap<N, IndexSet<N>> = IndexMap::new();
        for (parent, child) in edges {
            let values = match adj.entry(parent.clone()) {
                indexmap::map::Entry::Occupied(occupied_entry) => occupied_entry.into_mut(),
                indexmap::map::Entry::Vacant(vacant_entry) => {
                    nodes.insert(parent.clone());
                    vacant_entry.insert(IndexSet::new())
                }
            };
            if values.insert(child.clone()) {
                nodes.insert(child.clone());
            }
        }
        if nodes.len() > adj.len() {
            // Some nodes only appear as children, so we need to add them to the adj map
            // with empty adjacency sets.
            for node in nodes {
                adj.entry(node.clone()).or_default();
            }
        }
        Self { adj }
    }

    /// Constructs a new graph from an adjacency list. Iteration order is
    /// preserved from the input. Multi-edges in the input are removed.
    pub fn from_adjacency_list<I, AI>(adj_list: AI) -> Self
    where
        I: IntoIterator<Item = N>,
        AI: IntoIterator<Item = (N, I)>,
    {
        let mut nodes: IndexSet<N> = IndexSet::new();
        let mut adj: IndexMap<N, IndexSet<N>> = IndexMap::new();
        for (parent, children) in adj_list {
            let values = match adj.entry(parent.clone()) {
                indexmap::map::Entry::Occupied(occupied_entry) => occupied_entry.into_mut(),
                indexmap::map::Entry::Vacant(vacant_entry) => {
                    nodes.insert(parent.clone());
                    vacant_entry.insert(IndexSet::new())
                }
            };
            for child in children {
                if values.insert(child.clone()) {
                    nodes.insert(child.clone());
                }
            }
        }
        if nodes.len() > adj.len() {
            // Some nodes only appear as children, so we need to add them to the adj map
            // with empty adjacency sets.
            for node in nodes {
                adj.entry(node.clone()).or_default();
            }
        }
        Self { adj }
    }

    /// Returns a new graph with the same nodes as this graph and all edges
    /// reversed.
    pub fn reverse(&self) -> Self {
        let mut rev_adj: IndexMap<N, IndexSet<N>> = IndexMap::new();
        for (parent, children) in &self.adj {
            // Ensure parent is in rev_adj even if it has no children.
            rev_adj.entry(parent.clone()).or_default();
            for child in children {
                rev_adj
                    .entry(child.clone())
                    .or_default()
                    .insert(parent.clone());
            }
        }
        Self { adj: rev_adj }
    }
}

/// A FlowGraph is a directed graph with a designated start node.
///
/// Any node in the graph can be the start node. There are no reachability
/// requirements whatsoever: some nodes may be unreachable from the start node,
/// the start node could have incoming edges, the graph could be disconnected,
/// etc.
#[derive(Clone, Eq, PartialEq, Debug)]
pub struct FlowGraph<N>
where
    N: Clone + Eq + Hash + PartialEq,
{
    /// The graph.
    pub graph: SimpleDirectedGraph<N>,
    /// The start node.
    pub start_node: N,
}

/// Errors that be returned from FlowGraph APIs.
#[derive(Debug, Error, PartialEq)]
pub enum FlowGraphError {
    /// The start node was not found in the graph.
    #[error("Start node not found in graph")]
    StartNodeNotFound,
    /// The final node was not found in the graph.
    #[error("Final node not found in graph")]
    FinalNodeNotFound,
}

// An extended node in a flow graph. It can be either a "real" node, or a
// "virtual" node that we introduce to tag the start node.
#[derive(PartialEq, Eq, Clone, Hash)]
enum NodeTag<T> {
    Real(T),
    Virtual,
}

impl<N> FlowGraph<N>
where
    N: Clone + Eq + Hash + PartialEq,
{
    /// Constructs a new FlowGraph.
    pub fn new(graph: SimpleDirectedGraph<N>, start_node: N) -> Result<Self, FlowGraphError> {
        if !graph.contains_node(&start_node) {
            return Err(FlowGraphError::StartNodeNotFound);
        }
        Ok(Self { graph, start_node })
    }

    /// Creates a new FlowGraph by trimming the graph to the subgraph backwards
    /// reachable from the given final nodes, with the same start node as
    /// this FlowGraph (whether or not it is backwards reachable from the
    /// final nodes).
    pub fn trim(self, final_nodes: &[N]) -> Self {
        let mut rev_adj: HashMap<N, Vec<N>> = HashMap::new();
        for (parent, child) in self.graph.edges() {
            rev_adj
                .entry(child.clone())
                .or_default()
                .push(parent.clone());
        }

        let mut to_visit = VecDeque::with_capacity(final_nodes.len());
        let mut visited = HashSet::new();
        let mut adj: IndexMap<N, IndexSet<N>> = IndexMap::new();

        for final_node in final_nodes {
            visited.insert(final_node.clone());
            if *final_node != self.start_node {
                to_visit.push_back(final_node.clone());
            }
        }

        while let Some(child) = &to_visit.pop_front() {
            if let Some(parents) = rev_adj.get(child) {
                for parent in parents {
                    adj.entry(parent.clone()).or_default().insert(child.clone());
                    if visited.insert(parent.clone()) && *parent != self.start_node {
                        to_visit.push_back(parent.clone());
                    }
                }
            }
        }

        if visited.contains(&self.start_node)
            && let Some(start_node_parents) = rev_adj.get(&self.start_node)
        {
            for parent in start_node_parents {
                if visited.contains(parent) {
                    adj.entry(parent.clone())
                        .or_default()
                        .insert(self.start_node.clone());
                }
            }
        }

        // Ensure the start node is in the adj map, even if it has no edges in the
        // trimmed graph.
        adj.entry(self.start_node.clone()).or_default();

        Self::new(SimpleDirectedGraph::new(adj), self.start_node.clone()).unwrap()
    }

    /// Finds the closest common dominator to the final nodes in this FlowGraph,
    /// with the start node singled out as a virtual entry node.
    /// Returns None if there is no common dominator (this happens if and only
    /// if the start node is not a dominator of any of the final nodes).
    pub fn find_closest_common_dominator<NI>(&self, final_nodes: NI) -> Option<N>
    where
        NI: IntoIterator<Item = N>,
    {
        let extended_nodes = self
            .graph
            .nodes()
            .map(|n| NodeTag::Real(n.clone()))
            .chain(std::iter::once(NodeTag::Virtual));
        let extended_edges = self
            .graph
            .edges()
            .map(|(parent, child)| (NodeTag::Real(parent.clone()), NodeTag::Real(child.clone())))
            .chain(std::iter::once((
                NodeTag::Virtual,
                NodeTag::Real(self.start_node.clone()),
            )));
        let extended_targets = final_nodes
            .into_iter()
            .map(|final_node| NodeTag::Real(final_node.clone()));

        let result =
            find_closest_common_dominator(extended_nodes, extended_edges, extended_targets);

        match result {
            Ok(Some(NodeTag::Real(closest_common_dominator_value))) => {
                Some(closest_common_dominator_value.clone())
            }
            Ok(Some(NodeTag::Virtual)) => {
                // The virtual entry is the only common dominator, which means the start node is
                // not a common dominator of the final nodes.
                None
            }
            Ok(None) => {
                // No common dominator found. This should never happen.
                None
            }
            Err(_) => {
                // This should never happen since the extended graph is well-formed by
                // construction.
                None
            }
        }
    }
}

/// Let G be a FlowGraph with start node S. The ValueFlowGraph G' is a FlowGraph
/// derived from G and function value_fn. Let v(g) be the result of applying
/// value_fn to g. The nodes of G' are the set of values v(g), for all g in G.
/// For each edge g1->g2 in G, there is a corresponding edge v(g1)->v(g2) in G'.
/// The start node in G' is v(S). G' may have cycles even if G is a DAG.
#[derive(Clone, Eq, PartialEq)]
pub struct ValueFlowGraph<N, V>
where
    N: Clone + Eq + Hash + PartialEq,
    V: Clone + Eq + Hash + PartialEq,
{
    /// The flow graph of values.
    pub value_flow_graph: FlowGraph<V>,
    /// Maps nodes to their corresponding values.
    pub node_values: HashMap<N, V>,
    /// The inverse of node_values: maps each value to the nodes that have that
    /// value.
    pub value_to_nodes: HashMap<V, Vec<N>>,
}

struct ValueRecorder<'a, N, V, E> {
    node_values: HashMap<N, V>,
    value_to_nodes: HashMap<V, Vec<N>>,
    value_fn: &'a dyn Fn(&N) -> Result<V, E>,
}

impl<'a, N, V, E> ValueRecorder<'a, N, V, E>
where
    N: Hash + Eq + Clone,
    V: Hash + Eq + Clone,
{
    fn new(value_fn: &'a dyn Fn(&N) -> Result<V, E>) -> Self {
        Self {
            node_values: HashMap::new(),
            value_to_nodes: HashMap::new(),
            value_fn,
        }
    }

    fn get_value(&mut self, node: &N) -> Result<V, E> {
        match self.node_values.entry(node.clone()) {
            std::collections::hash_map::Entry::Occupied(entry) => Ok(entry.get().clone()),
            std::collections::hash_map::Entry::Vacant(entry) => {
                let value = entry.insert((self.value_fn)(node)?);
                self.value_to_nodes
                    .entry(value.clone())
                    .or_default()
                    .push(node.clone());
                Ok(value.clone())
            }
        }
    }

    fn done(self) -> (HashMap<N, V>, HashMap<V, Vec<N>>) {
        (self.node_values, self.value_to_nodes)
    }
}

impl<N, V> ValueFlowGraph<N, V>
where
    N: Hash + Eq + Clone,
    V: Hash + Eq + Clone,
{
    /// Let G be the FlowGraph represented by `g`. Constructs a ValueFlowGraph
    /// G' from G, using `value_fn` to obtain the value of nodes of G.
    ///
    /// Returns an error if value_fn returns an error.
    pub fn new<VF, E>(g: &FlowGraph<N>, value_fn: &VF) -> Result<Self, E>
    where
        VF: Fn(&N) -> Result<V, E>,
    {
        let mut value_recorder = ValueRecorder::new(value_fn);
        let mut value_adj: IndexMap<V, IndexSet<V>> = IndexMap::new();
        let mut seen_value_edge: HashSet<(V, V)> = HashSet::new();
        let mut seen_children: HashSet<N> = HashSet::new();

        let start_value = value_recorder.get_value(&g.start_node)?;

        for (parent, children) in &g.graph.adj {
            let parent_value = value_recorder.get_value(parent)?;
            let value_adj_entry = value_adj.entry(parent_value.clone()).or_default();
            seen_children.clear();
            for child in children {
                if seen_children.insert(child.clone()) {
                    let child_value = value_recorder.get_value(child)?;
                    if seen_value_edge.insert((parent_value.clone(), child_value.clone())) {
                        value_adj_entry.insert(child_value.clone());
                    }
                }
            }
        }

        let value_flow_graph =
            FlowGraph::new(SimpleDirectedGraph::new(value_adj), start_value).unwrap();
        let (node_values, value_to_nodes) = value_recorder.done();
        Ok(Self {
            value_flow_graph,
            node_values,
            value_to_nodes,
        })
    }

    /// Finds the dominator value for this ValueFlowGraph, if there is one.
    /// The dominator value is the closes common dominator to the values of
    /// the final nodes in the ValueFlowGraph, with the value of the
    /// start node singled out as a virtual entry node.
    ///
    /// The dominator value is guaranteed to exist if the start node is a
    /// dominator of the final nodes in G.
    pub fn find_dominator_value(&self, final_nodes: &[N]) -> Result<Option<V>, FlowGraphError> {
        let final_values: Vec<_> = final_nodes
            .iter()
            .map(|final_node| match self.node_values.get(final_node) {
                Some(value) => Ok(value.clone()),
                None => Err(FlowGraphError::FinalNodeNotFound),
            })
            .try_collect()?;
        Ok(self
            .value_flow_graph
            .find_closest_common_dominator(final_values))
    }
}

#[cfg(test)]
mod tests {
    use indexmap::indexmap;
    use indexmap::indexset;

    use super::*;

    fn new_dominator_finder(nodes: Vec<&str>, edges: Vec<(&str, &str)>) -> DominatorFinder<String> {
        let nodes_string: Vec<String> = nodes.iter().map(|&n| n.to_string()).collect();
        let edges_string: Vec<(String, String)> = edges
            .iter()
            .map(|&(u, v)| (u.to_string(), v.to_string()))
            .collect();
        DominatorFinder::new(nodes_string, edges_string).unwrap()
    }

    fn closest_common_dominator(
        nodes: &[&str],
        edges: &[(&str, &str)],
        s: Vec<&str>,
        expected: Option<&str>,
    ) {
        let finder = new_dominator_finder(nodes.to_owned(), edges.to_owned());
        let s_string = s.iter().map(|&n| n.to_string()).collect_vec();
        let result = finder.closest_common_dominator(s_string).unwrap();
        assert_eq!(result, expected.map(|e| e.to_string()));
    }

    fn closest_common_dominator_expect_error(
        nodes: &[&str],
        edges: &[(&str, &str)],
        s: Vec<&str>,
        expected_error: DominatorFinderError,
    ) {
        let finder = new_dominator_finder(nodes.to_owned(), edges.to_owned());
        let s_string = s.iter().map(|&n| n.to_string()).collect_vec();
        let result = finder.closest_common_dominator(s_string);
        assert_eq!(result, Err(expected_error));
    }

    #[test]
    fn test_closest_common_dominator_split() {
        //   /-> B -> D
        // A
        //   \-> C -> D
        let nodes = vec!["A", "B", "C", "D"];
        let edges = vec![("A", "B"), ("A", "C"), ("B", "D"), ("C", "D")];

        closest_common_dominator(&nodes, &edges, vec!["A"], Some("A"));
        closest_common_dominator(&nodes, &edges, vec!["B"], Some("B"));
        closest_common_dominator(&nodes, &edges, vec!["C"], Some("C"));
        closest_common_dominator(&nodes, &edges, vec!["D"], Some("D"));

        closest_common_dominator(&nodes, &edges, vec!["B", "C"], Some("A"));
        closest_common_dominator(&nodes, &edges, vec!["B", "D"], Some("A"));
        closest_common_dominator(&nodes, &edges, vec!["B", "C", "D"], Some("A"));
    }

    #[test]
    fn test_closest_common_dominator_linear_chain() {
        // A -> B -> C -> D
        let nodes = vec!["A", "B", "C", "D"];
        let edges = vec![("A", "B"), ("B", "C"), ("C", "D")];

        closest_common_dominator(&nodes, &edges, vec!["A"], Some("A"));
        closest_common_dominator(&nodes, &edges, vec!["B"], Some("B"));
        closest_common_dominator(&nodes, &edges, vec!["C"], Some("C"));
        closest_common_dominator(&nodes, &edges, vec!["D"], Some("D"));

        closest_common_dominator(&nodes, &edges, vec!["A", "B"], Some("A"));
        closest_common_dominator(&nodes, &edges, vec!["A", "C"], Some("A"));
        closest_common_dominator(&nodes, &edges, vec!["A", "D"], Some("A"));
        closest_common_dominator(&nodes, &edges, vec!["B", "D"], Some("B"));
        closest_common_dominator(&nodes, &edges, vec!["C", "D"], Some("C"));
        closest_common_dominator(&nodes, &edges, vec!["A", "B", "C", "D"], Some("A"));
    }

    #[test]
    fn test_closest_common_dominator_disjoint_no_common() {
        // A -> B
        // C -> D
        let nodes = vec!["A", "B", "C", "D"];
        let edges = vec![("A", "B"), ("C", "D")];

        closest_common_dominator(&nodes, &edges, vec!["A", "C"], None);
        closest_common_dominator(&nodes, &edges, vec!["A", "D"], None);
        closest_common_dominator(&nodes, &edges, vec!["B", "D"], None);

        closest_common_dominator(&nodes, &edges, vec!["A"], Some("A"));
        closest_common_dominator(&nodes, &edges, vec!["B"], Some("B"));
        closest_common_dominator(&nodes, &edges, vec!["A", "B"], Some("A"));
    }

    #[test]
    fn test_closest_common_dominator_classic_diamond() {
        //      /-> B -\
        //    A          -> D -> E
        //      \-> C -/
        let nodes = vec!["A", "B", "C", "D", "E"];
        let edges = vec![("A", "B"), ("A", "C"), ("B", "D"), ("C", "D"), ("D", "E")];

        closest_common_dominator(&nodes, &edges, vec!["B", "C"], Some("A"));
        closest_common_dominator(&nodes, &edges, vec!["B", "E"], Some("A"));
        closest_common_dominator(&nodes, &edges, vec!["D"], Some("D"));
        closest_common_dominator(&nodes, &edges, vec!["D", "E"], Some("D"));
        closest_common_dominator(&nodes, &edges, vec!["A", "D"], Some("A"));
    }

    #[test]
    fn test_closest_common_dominator_basic_y_shape() {
        // A
        //  \
        //    --> C -> D
        //  /
        // B
        let nodes = vec!["A", "B", "C", "D"];
        let edges = vec![("A", "C"), ("B", "C"), ("C", "D")];

        closest_common_dominator(&nodes, &edges, vec!["A", "B"], None);
        closest_common_dominator(&nodes, &edges, vec!["A", "C"], None);
        closest_common_dominator(&nodes, &edges, vec!["C", "D"], Some("C"));

        closest_common_dominator(&nodes, &edges, vec!["A"], Some("A"));
        closest_common_dominator(&nodes, &edges, vec!["B"], Some("B"));
        closest_common_dominator(&nodes, &edges, vec!["C"], Some("C"));
        closest_common_dominator(&nodes, &edges, vec!["D"], Some("D"));
    }

    #[test]
    fn test_closest_common_dominator_single_node() {
        // A
        let nodes = vec!["A"];
        let edges = vec![];

        closest_common_dominator(&nodes, &edges, vec!["A"], Some("A"));
    }

    #[test]
    fn test_closest_common_dominator_generic_integers() {
        // Using Integers instead of Strings
        // 1 -> 2
        // 1 -> 3
        let nodes = vec![1, 2, 3];
        let edges = vec![(1, 2), (1, 3)];

        let finder = DominatorFinder::new(nodes, edges).unwrap();
        let result = finder.closest_common_dominator([2, 3]);
        assert_eq!(result, Ok(Some(1)));
    }

    #[test]
    fn test_closest_common_dominator_complex_multi_source_multi_sink() {
        //       /-> E
        // A -> B
        //       \-> F
        //           ^
        //           |
        // C --> D --/
        let nodes = vec!["A", "B", "C", "D", "E", "F"];
        let edges = vec![("A", "B"), ("B", "E"), ("B", "F"), ("C", "D"), ("D", "F")];

        closest_common_dominator(&nodes, &edges, vec!["E", "F"], None);
        closest_common_dominator(&nodes, &edges, vec!["F"], Some("F"));
        closest_common_dominator(&nodes, &edges, vec!["B", "F"], None);
    }

    #[test]
    fn test_closest_common_dominator_simple_cycle_with_entry() {
        //
        // A -> B -> C -> D
        //      ^         |
        //      |         |
        //      \--------/

        let nodes = vec!["A", "B", "C", "D"];
        let edges = vec![("A", "B"), ("B", "C"), ("C", "D"), ("D", "B")];

        closest_common_dominator(&nodes, &edges, vec!["A", "B"], Some("A"));
        closest_common_dominator(&nodes, &edges, vec!["A", "C"], Some("A"));
        closest_common_dominator(&nodes, &edges, vec!["A", "B", "C"], Some("A"));
        closest_common_dominator(&nodes, &edges, vec!["B", "C"], Some("B"));
        closest_common_dominator(&nodes, &edges, vec!["B", "C", "D"], Some("B"));

        closest_common_dominator(&nodes, &edges, vec!["A"], Some("A"));
        closest_common_dominator(&nodes, &edges, vec!["B"], Some("B"));
        closest_common_dominator(&nodes, &edges, vec!["C"], Some("C"));
        closest_common_dominator(&nodes, &edges, vec!["D"], Some("D"));
    }

    #[test]
    fn test_closest_common_dominator_figure_eight_with_bridge() {
        //
        //  A -> B -> C -> D -> E -> F -> G
        //       ^         |    ^         |
        //       |         |    |         |
        //        \_______/      \_______/
        let nodes = vec!["A", "B", "C", "D", "E", "F", "G"];
        let edges = vec![
            ("A", "B"), // entry
            ("B", "C"),
            ("C", "D"),
            ("D", "B"), // Loop 1
            ("D", "E"), // Bridge
            ("E", "F"),
            ("F", "G"),
            ("G", "E"), // Loop 2
        ];

        closest_common_dominator(&nodes, &edges, vec!["B", "C"], Some("B"));
        closest_common_dominator(&nodes, &edges, vec!["B", "D"], Some("B"));
        closest_common_dominator(&nodes, &edges, vec!["B", "E"], Some("B"));
        closest_common_dominator(&nodes, &edges, vec!["C", "E"], Some("C"));
        closest_common_dominator(&nodes, &edges, vec!["C", "F"], Some("C"));
        closest_common_dominator(&nodes, &edges, vec!["D", "E"], Some("D"));
        closest_common_dominator(&nodes, &edges, vec!["D", "F"], Some("D"));
        closest_common_dominator(&nodes, &edges, vec!["E", "G"], Some("E"));
        closest_common_dominator(&nodes, &edges, vec!["F", "G"], Some("F"));
    }

    #[test]
    fn test_closest_common_dominator_figure_eight() {
        //
        //  A -> B -> C --> D   -> E -> F
        //       ^         | ^          |
        //       |         | |          |
        //        \_______/  \_________/
        let nodes = vec!["A", "B", "C", "D", "E", "F"];
        let edges = vec![
            ("A", "B"), // entry
            ("B", "C"),
            ("C", "D"),
            ("D", "B"), // Loop 1
            ("D", "E"),
            ("E", "F"),
            ("F", "D"), // Loop 2
        ];

        closest_common_dominator(&nodes, &edges, vec!["B", "C"], Some("B"));
        closest_common_dominator(&nodes, &edges, vec!["B", "D"], Some("B"));
        closest_common_dominator(&nodes, &edges, vec!["B", "E"], Some("B"));
        closest_common_dominator(&nodes, &edges, vec!["C", "D"], Some("C"));
        closest_common_dominator(&nodes, &edges, vec!["C", "E"], Some("C"));
        closest_common_dominator(&nodes, &edges, vec!["D", "E"], Some("D"));
        closest_common_dominator(&nodes, &edges, vec!["D", "F"], Some("D"));
        closest_common_dominator(&nodes, &edges, vec!["E", "F"], Some("E"));
    }

    #[test]
    fn test_closest_common_dominator_entry_cycle_dominance() {
        // B -> C -> B (Loop)
        // A -> B
        let nodes = vec!["A", "B", "C"];
        let edges = vec![("A", "B"), ("B", "C"), ("C", "B")];

        closest_common_dominator(&nodes, &edges, vec!["A"], Some("A"));
        closest_common_dominator(&nodes, &edges, vec!["B"], Some("B"));
        closest_common_dominator(&nodes, &edges, vec!["C"], Some("C"));

        closest_common_dominator(&nodes, &edges, vec!["A", "B"], Some("A"));
        closest_common_dominator(&nodes, &edges, vec!["A", "C"], Some("A"));
        closest_common_dominator(&nodes, &edges, vec!["B", "C"], Some("B"));
        closest_common_dominator(&nodes, &edges, vec!["A", "B", "C"], Some("A"));
    }

    #[test]
    fn test_closest_common_dominator_nested_loops() {
        //           /---> E
        //           |     |
        //           |     |
        // A -> B -> C <--/
        //      ^     \--> D
        //      |          |
        //      |----------|
        let nodes = vec!["A", "B", "C", "D", "E"];
        let edges = vec![
            ("A", "B"),
            ("B", "C"),
            ("C", "D"),
            ("C", "E"),
            ("E", "C"),
            ("D", "B"),
        ];

        closest_common_dominator(&nodes, &edges, vec!["A", "B"], Some("A"));
        closest_common_dominator(&nodes, &edges, vec!["A", "C"], Some("A"));
        closest_common_dominator(&nodes, &edges, vec!["B", "C"], Some("B"));
        closest_common_dominator(&nodes, &edges, vec!["B", "D"], Some("B"));
        closest_common_dominator(&nodes, &edges, vec!["B", "E"], Some("B"));
        closest_common_dominator(&nodes, &edges, vec!["C", "D"], Some("C"));
        closest_common_dominator(&nodes, &edges, vec!["C", "E"], Some("C"));
        closest_common_dominator(&nodes, &edges, vec!["D", "E"], Some("C"));

        closest_common_dominator(&nodes, &edges, vec!["B", "C", "D"], Some("B"));
        closest_common_dominator(&nodes, &edges, vec!["B", "C", "E"], Some("B"));
        closest_common_dominator(&nodes, &edges, vec!["B", "D", "E"], Some("B"));
        closest_common_dominator(&nodes, &edges, vec!["C", "D", "E"], Some("C"));

        closest_common_dominator(&nodes, &edges, vec!["B", "C", "D", "E"], Some("B"));
    }

    #[test]
    fn test_closest_common_dominator_tree() {
        // A -> B -> C
        // \     \-> D
        //  \------> E
        let nodes = vec!["A", "B", "C", "D", "E"];
        let edges = vec![("A", "B"), ("B", "C"), ("B", "D"), ("A", "E")];

        closest_common_dominator(&nodes, &edges, vec!["B", "C"], Some("B"));
        closest_common_dominator(&nodes, &edges, vec!["B", "E"], Some("A"));
        closest_common_dominator(&nodes, &edges, vec!["C", "D"], Some("B"));
        closest_common_dominator(&nodes, &edges, vec!["C", "E"], Some("A"));

        closest_common_dominator(&nodes, &edges, vec!["B", "C", "D"], Some("B"));
        closest_common_dominator(&nodes, &edges, vec!["C", "D", "E"], Some("A"));
    }

    #[test]
    fn test_closest_common_dominator_bypassing_path() {
        // A -> B -> C -> D
        // |              ^
        // v              |
        // E -------------/
        let nodes = vec!["A", "B", "C", "D", "E"];
        let edges = vec![("A", "B"), ("B", "C"), ("C", "D"), ("A", "E"), ("E", "D")];

        closest_common_dominator(&nodes, &edges, vec!["B", "C"], Some("B"));
        closest_common_dominator(&nodes, &edges, vec!["B", "D"], Some("A"));
        closest_common_dominator(&nodes, &edges, vec!["B", "E"], Some("A"));
        closest_common_dominator(&nodes, &edges, vec!["C", "D"], Some("A"));
        closest_common_dominator(&nodes, &edges, vec!["C", "E"], Some("A"));
        closest_common_dominator(&nodes, &edges, vec!["D", "E"], Some("A"));

        closest_common_dominator(&nodes, &edges, vec!["B", "C", "D"], Some("A"));
        closest_common_dominator(&nodes, &edges, vec!["C", "D", "E"], Some("A"));
    }

    #[test]
    fn test_closest_common_dominator_infinite_loop_trap() {
        // A->B, C->D->C (Trap)
        let nodes = vec!["A", "B", "C", "D"];
        let edges = vec![
            ("A", "B"), // Safe path
            ("C", "D"),
            ("D", "C"), // Trap
        ];

        closest_common_dominator(&nodes, &edges, vec!["A", "C"], None);
        closest_common_dominator(&nodes, &edges, vec!["B", "C"], None);
        closest_common_dominator(&nodes, &edges, vec!["C", "D"], None);
    }

    #[test]
    fn test_closest_common_dominator_self_loop_handling() {
        // A->A (Self loop), A->B
        let nodes = vec!["A", "B"];
        let edges = vec![("A", "A"), ("A", "B")];
        closest_common_dominator(&nodes, &edges, vec!["A"], Some("A"));
    }

    #[test]
    fn test_closest_common_dominator_multi_edge() {
        // Shape: A->B (x2), B->C.
        let nodes = vec!["A", "B", "C"];
        let edges = vec![
            ("A", "B"),
            ("A", "B"), // Duplicate edge
            ("B", "C"),
        ];
        closest_common_dominator(&nodes, &edges, vec!["A"], Some("A"));
    }

    #[test]
    fn test_closest_common_dominator_empty_target_set() {
        // A -> B
        let nodes = vec!["A", "B"];
        let edges = vec![("A", "B")];
        closest_common_dominator(&nodes, &edges, vec![], None);
    }

    #[test]
    fn test_closest_common_dominator_empty_graph() {
        let nodes: Vec<String> = vec![];
        let edges: Vec<(String, String)> = vec![];
        assert_eq!(
            DominatorFinder::new(nodes, edges).err(),
            Some(DominatorFinderError::GraphHasNoEntryNode)
        );
    }

    #[test]
    fn test_closest_common_dominator_repeated_node() {
        // A -> B
        let nodes = vec!["A", "B", "A", "B"];
        let edges = vec![("A", "B")];

        closest_common_dominator(&nodes, &edges, vec!["A"], Some("A"));
        closest_common_dominator(&nodes, &edges, vec!["B"], Some("B"));
        closest_common_dominator(&nodes, &edges, vec!["A", "B"], Some("A"));
    }

    #[test]
    fn test_closest_common_dominator_invalid_edge() {
        let nodes = vec!["A", "B"];
        {
            let edges = vec![("A", "C")];
            let finder = DominatorFinder::new(nodes.clone(), edges);
            assert_eq!(
                finder.err(),
                Some(DominatorFinderError::EdgeContainsUnknownNode)
            );
        }
        {
            let edges = vec![("C", "A")];
            let finder = DominatorFinder::new(nodes.clone(), edges);
            assert_eq!(
                finder.err(),
                Some(DominatorFinderError::EdgeContainsUnknownNode)
            );
        }
        {
            let edges = vec![("C", "D")];
            let finder = DominatorFinder::new(nodes.clone(), edges);
            assert_eq!(
                finder.err(),
                Some(DominatorFinderError::EdgeContainsUnknownNode)
            );
        }
    }

    #[test]
    fn test_closest_common_dominator_invalid_node_in_target() {
        // A -> B
        let nodes = vec!["A", "B"];
        let edges = vec![("A", "B")];

        closest_common_dominator_expect_error(
            &nodes,
            &edges,
            vec!["C"],
            DominatorFinderError::TargetSetContainsUnknownNode,
        );
        closest_common_dominator_expect_error(
            &nodes,
            &edges,
            vec!["A", "C"],
            DominatorFinderError::TargetSetContainsUnknownNode,
        );
        closest_common_dominator_expect_error(
            &nodes,
            &edges,
            vec!["B", "C"],
            DominatorFinderError::TargetSetContainsUnknownNode,
        );
    }

    #[test]
    fn test_simple_directed_graph_new() {
        let adj = indexmap! {
            "A" => indexset! {"B"},
            "B" => indexset!{},
        };
        let graph = SimpleDirectedGraph::new(adj.clone());
        assert_eq!(graph.adj, adj);

        // adj does not have entries for "B" or "D".
        let adj = indexmap! {
            "A" => indexset! {"B", "C", "D"},
            "C" => indexset!{},
        };
        let graph = SimpleDirectedGraph::new(adj);
        assert_eq!(
            graph.adj,
            indexmap! {
                "A" => indexset! {"B", "C", "D"},
                "C" => indexset!{},
                "B" => indexset!{},
                "D" => indexset!{},
            }
        );
    }

    #[test]
    fn test_simple_directed_graph_nodes() {
        let graph = SimpleDirectedGraph::from_edge_list(vec![("A", "B"), ("B", "C")]);
        let nodes = graph.nodes().copied().collect_vec();
        assert_eq!(nodes, vec!["A", "B", "C"]);

        let graph = SimpleDirectedGraph::<String>::from_edge_list(vec![]);
        let nodes = graph.nodes().cloned().collect_vec();
        assert!(nodes.is_empty());
    }

    #[test]
    fn test_simple_directed_graph_edges() {
        let graph = SimpleDirectedGraph::from_edge_list(vec![("A", "B"), ("B", "C"), ("A", "C")]);
        let edges = graph.edges().map(|(&u, &v)| (u, v)).collect_vec();
        assert_eq!(edges, vec![("A", "B"), ("A", "C"), ("B", "C")]);

        let graph = SimpleDirectedGraph::<String>::from_edge_list(vec![]);
        let edges = graph.edges().collect_vec();
        assert!(edges.is_empty());
    }

    #[test]
    fn test_simple_directed_graph_adjacent_nodes() {
        let graph = SimpleDirectedGraph::from_edge_list(vec![("A", "B"), ("A", "C"), ("B", "D")]);
        assert_eq!(
            graph.adjacent_nodes(&"A").unwrap().copied().collect_vec(),
            vec!["B", "C"]
        );
        assert_eq!(
            graph.adjacent_nodes(&"B").unwrap().copied().collect_vec(),
            vec!["D"]
        );
        assert!(graph.adjacent_nodes(&"C").unwrap().next().is_none());
        assert!(graph.adjacent_nodes(&"Z").is_none());
    }

    #[test]
    fn test_simple_directed_graph_contains_node() {
        let graph = SimpleDirectedGraph::from_edge_list(vec![("A", "B"), ("B", "C")]);
        assert!(graph.contains_node(&"A"));
        assert!(graph.contains_node(&"B"));
        assert!(graph.contains_node(&"C"));
        assert!(!graph.contains_node(&"D"));
    }

    #[test]
    fn test_simple_directed_graph_from_edge_list() {
        let graph = SimpleDirectedGraph::from_edge_list(vec![
            ("A", "B"),
            ("A", "C"),
            ("B", "C"),
            ("A", "B"),
        ]);
        let nodes = graph.nodes().copied().collect_vec();
        assert_eq!(nodes, vec!["A", "B", "C"]);
        let edges = graph.edges().map(|(&u, &v)| (u, v)).collect_vec();
        assert_eq!(edges, vec![("A", "B"), ("A", "C"), ("B", "C")]);

        let graph = SimpleDirectedGraph::from_edge_list(vec![("B", "C"), ("A", "B")]);
        let nodes = graph.nodes().copied().collect_vec();
        assert_eq!(nodes, vec!["B", "A", "C"]);
        let edges = graph.edges().map(|(&u, &v)| (u, v)).collect_vec();
        assert_eq!(edges, vec![("B", "C"), ("A", "B")]);
    }

    #[test]
    fn test_simple_directed_graph_from_adjacency_list() {
        let adj_list = vec![("A", vec!["B", "C"]), ("B", vec!["D"]), ("C", vec![])];
        let graph = SimpleDirectedGraph::from_adjacency_list(adj_list);
        let nodes = graph.nodes().copied().collect_vec();
        assert_eq!(nodes, vec!["A", "B", "C", "D"]);
        let edges = graph.edges().map(|(&u, &v)| (u, v)).collect_vec();
        assert_eq!(edges, vec![("A", "B"), ("A", "C"), ("B", "D")]);

        let adj_list_with_missing = vec![("A", vec!["B"])];
        let graph = SimpleDirectedGraph::from_adjacency_list(adj_list_with_missing);
        assert!(graph.contains_node(&"B"));
    }

    #[test]
    fn test_flow_graph_new() {
        let graph = SimpleDirectedGraph::from_edge_list(vec![("A", "B")]);
        let flow_graph = FlowGraph::new(graph.clone(), "A").unwrap();
        assert_eq!(flow_graph.graph, graph);
        assert_eq!(flow_graph.start_node, "A");

        let flow_graph_err = FlowGraph::new(graph, "C");
        assert_eq!(
            flow_graph_err.err(),
            Some(FlowGraphError::StartNodeNotFound)
        );
    }

    #[test]
    fn test_flow_graph_trim() {
        // A -> B -> C -> D -> E
        // |         ^
        // v         |
        // F --------/
        let edges = vec![
            ("A", "B"),
            ("B", "C"),
            ("C", "D"),
            ("D", "E"),
            ("A", "F"),
            ("F", "C"),
        ];
        let simple_graph = SimpleDirectedGraph::from_edge_list(edges.clone());
        let flow_graph = FlowGraph::new(simple_graph, "A").unwrap();

        // Trim from E
        let trimmed = flow_graph.clone().trim(&["E"]);
        let expected_graph = SimpleDirectedGraph::from_edge_list(edges.clone());
        assert_eq!(trimmed.graph, expected_graph);
        assert_eq!(trimmed.start_node, "A");

        // Trim from C and E
        let trimmed = flow_graph.clone().trim(&["C", "E"]);
        let expected_graph = SimpleDirectedGraph::from_edge_list(edges.clone());
        assert_eq!(trimmed.graph, expected_graph);

        // TODO: wrong comment, make sure there is a case for "start node is not
        // reachable" and "start node is in target set" Trim from F (start node
        // is not reachable)
        let trimmed = flow_graph.clone().trim(&["F"]);
        let expected_graph = SimpleDirectedGraph::from_edge_list(vec![("A", "F")]);
        assert_eq!(trimmed.graph, expected_graph);

        // Trim from A (start node is in the target set)
        let trimmed = flow_graph.clone().trim(&["A"]);
        let expected_graph = SimpleDirectedGraph::new(indexmap! {"A" => indexset! {}});
        assert_eq!(trimmed.graph, expected_graph);
    }

    #[test]
    fn test_flow_graph_find_closest_common_dominator() {
        // A -> B -> C -> D
        let edges = vec![("A", "B"), ("B", "C"), ("C", "D")];
        let simple_graph = SimpleDirectedGraph::from_edge_list(edges);
        let flow_graph = FlowGraph::new(simple_graph, "A").unwrap();
        assert_eq!(
            flow_graph.find_closest_common_dominator(vec!["C", "D"]),
            Some("C")
        );
        assert_eq!(
            flow_graph.find_closest_common_dominator(vec!["B", "D"]),
            Some("B")
        );
        assert_eq!(
            flow_graph.find_closest_common_dominator(vec!["A", "D"]),
            Some("A")
        );

        // Diamond: A -> {B, C} -> D
        let edges = vec![("A", "B"), ("A", "C"), ("B", "D"), ("C", "D")];
        let simple_graph = SimpleDirectedGraph::from_edge_list(edges);
        let flow_graph = FlowGraph::new(simple_graph, "A").unwrap();
        assert_eq!(
            flow_graph.find_closest_common_dominator(vec!["B", "C"]),
            Some("A")
        );
        assert_eq!(
            flow_graph.find_closest_common_dominator(vec!["B", "D"]),
            Some("A")
        );
        assert_eq!(
            flow_graph.find_closest_common_dominator(vec!["D"]),
            Some("D")
        );

        // Disjoint: A -> B, C -> D
        let edges = vec![("A", "B"), ("C", "D")];
        let simple_graph = SimpleDirectedGraph::from_edge_list(edges);
        let flow_graph = FlowGraph::new(simple_graph, "A").unwrap();
        assert_eq!(
            flow_graph.find_closest_common_dominator(vec!["B", "D"]),
            None
        );
        assert_eq!(
            flow_graph.find_closest_common_dominator(vec!["B", "C"]),
            None
        );
    }

    #[test]
    fn test_value_flow_graph_new() {
        // A(1) -> B(1) -> C(2)
        let edges = vec![("A", "B"), ("B", "C")];
        let simple_graph = SimpleDirectedGraph::from_edge_list(edges);
        let flow_graph = FlowGraph::new(simple_graph, "A").unwrap();
        let value_fn = |node: &&str| -> Result<i32, ()> {
            if *node == "A" || *node == "B" {
                Ok(1)
            } else {
                Ok(2)
            }
        };
        let value_flow_graph = ValueFlowGraph::new(&flow_graph, &value_fn).unwrap();

        let expected_value_adj: IndexMap<i32, IndexSet<i32>> =
            IndexMap::from([(1, IndexSet::from([1, 2])), (2, IndexSet::new())]);
        let expected_flow_graph =
            FlowGraph::new(SimpleDirectedGraph::new(expected_value_adj), 1).unwrap();
        assert_eq!(value_flow_graph.value_flow_graph, expected_flow_graph);

        let mut expected_node_values = HashMap::new();
        expected_node_values.insert("A", 1);
        expected_node_values.insert("B", 1);
        expected_node_values.insert("C", 2);
        assert_eq!(value_flow_graph.node_values, expected_node_values);

        let mut expected_value_to_nodes = HashMap::new();
        expected_value_to_nodes.insert(1, vec!["A", "B"]);
        expected_value_to_nodes.insert(2, vec!["C"]);
        assert_eq!(value_flow_graph.value_to_nodes, expected_value_to_nodes);

        // Test value_fn error
        let value_fn_err = |_: &&str| -> Result<i32, String> { Err("Error".to_string()) };
        let value_flow_graph_err = ValueFlowGraph::new(&flow_graph, &value_fn_err);
        assert_eq!(value_flow_graph_err.err(), Some("Error".to_string()));
    }

    #[test]
    fn test_value_flow_graph_find_dominator_value() {
        // A(1) -> B(1) -> C(2) -> D(3)
        //          \------------> E(3)
        let edges = vec![("A", "B"), ("B", "C"), ("C", "D"), ("B", "E")];
        let simple_graph = SimpleDirectedGraph::from_edge_list(edges);
        let flow_graph = FlowGraph::new(simple_graph, "A").unwrap();
        let value_fn = |node: &&str| match *node {
            "A" | "B" => Ok(1),
            "C" => Ok(2),
            "D" | "E" => Ok(3),
            _ => Err("Unknown node".to_string()),
        };
        let value_flow_graph = ValueFlowGraph::new(&flow_graph, &value_fn).unwrap();

        // Value graph (* means node has a self-loop):
        //   1* -> 2 -> 3
        //    \         ^
        //     \--------|
        assert_eq!(
            value_flow_graph.find_dominator_value(&["D", "E"]),
            Ok(Some(3))
        );
        assert_eq!(
            value_flow_graph.find_dominator_value(&["C", "D"]),
            Ok(Some(1))
        );
        assert_eq!(
            value_flow_graph.find_dominator_value(&["B", "C"]),
            Ok(Some(1))
        );

        // Disjoint values: A(1) -> B(1), C(2) -> D(2)
        let edges = vec![("A", "B"), ("C", "D")];
        let simple_graph = SimpleDirectedGraph::from_edge_list(edges);
        let flow_graph = FlowGraph::new(simple_graph, "A").unwrap();
        let value_fn = |node: &&str| match *node {
            "A" | "B" => Ok(1),
            "C" | "D" => Ok(2),
            _ => Err("Unknown node".to_string()),
        };
        let value_flow_graph = ValueFlowGraph::new(&flow_graph, &value_fn).unwrap();
        assert_eq!(value_flow_graph.find_dominator_value(&["B", "D"]), Ok(None));
    }
}
