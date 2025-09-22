import networkx as nx
from collections import deque
import walker


def build_graph(graph: list, undirected = False) -> nx.DiGraph | nx.Graph:
    if undirected:
        G = nx.Graph()
    else:
        G = nx.DiGraph()
    for triplet in graph:
        h, r, t = triplet
        G.add_edge(h.strip(), t.strip(), relation=r.strip())
    return G


def dfs(graph, start_node_list, max_length):
    """
    Find all paths within max_length starting from start_node_list in graph using DFS.

    Args:
        graph (nx.DiGraph): Directed graph
        start_node (List[str]): A list of start nodes
        max_length (int): Maximum length of path

    Returns:
        List[List[tuple]]: Find paths
    """
    def dfs_visit(node, path):
        if len(path) > max_length:
            return
        try:
            for neighbor in graph.neighbors(node):
                rel = graph[node][neighbor]["relation"]
                new_path = path + [(node, rel, neighbor)]
                if len(new_path) <= max_length:
                    path_lists.add(tuple(new_path))
                dfs_visit(neighbor, new_path)
        except Exception as e:
            print(e)
            pass

    path_lists = set()
    for start_node in start_node_list:
        dfs_visit(start_node, [])

    return list(path_lists)


def bfs_with_rule(graph, start_node, target_rule, max_p=10):
    result_paths = []
    queue = deque([(start_node, [])])
    while queue:
        current_node, current_path = queue.popleft()

        if len(current_path) == len(target_rule):
            result_paths.append(current_path)

        if len(current_path) < len(target_rule):
            if current_node not in graph:
                continue
            for neighbor in graph.neighbors(current_node):
                rel = graph[current_node][neighbor]["relation"]
                if rel != target_rule[len(current_path)] or len(current_path) > len(
                    target_rule
                ):
                    continue
                queue.append((neighbor, current_path + [(current_node, rel, neighbor)]))

    return result_paths


def get_truth_paths(q_entity: list, a_entity: list, graph: nx.Graph) -> list:
    """
    Get shortest paths connecting question and answer entities.
    """
    # Select paths
    paths = []
    for h in q_entity:
        if h not in graph:
            continue
        for t in a_entity:
            if t not in graph:
                continue
            try:
                for p in nx.all_shortest_paths(graph, h, t):
                    paths.append(p)
            except:
                pass
    # Add relation to paths
    result_paths = []
    for p in paths:
        tmp = []
        for i in range(len(p) - 1):
            u = p[i]
            v = p[i + 1]
            tmp.append((u, graph[u][v]["relation"], v))
        result_paths.append(tmp)
    return result_paths


def get_simple_paths(q_entity: list, a_entity: list, graph: nx.Graph, hop=2) -> list:
    """
    Get all simple paths connecting question and answer entities within given hop
    """
    # Select paths
    paths = []
    for h in q_entity:
        if h not in graph:
            continue
        for t in a_entity:
            if t not in graph:
                continue
            try:
                for p in nx.all_simple_edge_paths(graph, h, t, cutoff=hop):
                    paths.append(p)
            except:
                pass
    # Add relation to paths
    result_paths = []
    for p in paths:
        result_paths.append([(e[0], graph[e[0]][e[1]]["relation"], e[1]) for e in p])
    return result_paths

