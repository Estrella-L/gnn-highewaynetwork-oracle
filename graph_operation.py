

def graph_depth(graph, starting_vertex):
    visited = list()
    queue = list()
    visit_count = 0
    depth = 0
    nodes = graph[0]
    num_nodes = len(nodes)
    neigh_info = graph[5]
    queue.append(starting_vertex)
    visited.append(starting_vertex)
    visit_count += 1
    while visit_count<num_nodes:
        current_node = queue.pop(0)
        current_neighbors = neigh_info[current_node]
        for u in current_neighbors:
            if u not in visited:
                queue.append(u)
                visited.append(u)
                visit_count += 1
        depth+=1
    return depth