from queue import PriorityQueue


def djikstra(graph, initial, target):
    visited = {initial: 0}
    path = {}
    path[initial] = None
    nodes = set(graph.keys())

    while nodes:
        current_node = min(nodes, key=lambda node: visited.get(node, float("inf")))
        nodes.remove(current_node)
        if current_node == target:
            break

        for neighbor, weight in graph[current_node].items():
            if (
                neighbor not in visited
                or visited[current_node] + weight < visited[neighbor]
            ):
                visited[neighbor] = visited[current_node] + weight
                path[neighbor] = current_node

    return visited, path


graph = {
    "A": {"B": 1, "C": 4},
    "B": {"C": 2, "D": 5},
    "C": {"D": 3},
    "D": {"B": 1, "E": 1},
    "E": {"C": 1},
}

initial = "A"
target = "E"

visited, path = djikstra(graph, initial, target)

# print shortest distance from initial to target
print("Shortest distance: ", visited[target])

# print the shortest path from initial to target
current_node = target
while current_node is not None:
    print(current_node, end=" ")
    current_node = path[current_node]
    if current_node is None:
        break
print(initial)
