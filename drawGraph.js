const graphData = {
  A: [
    { node: "B", weight: 1 },
    { node: "C", weight: 4 },
    { node: "D", weight: 3 },
  ],
  B: [
    { node: "A", weight: 1 },
    { node: "C", weight: 2 },
    { node: "D", weight: 5 },
    { node: "E", weight: 8 },
  ],
  C: [
    { node: "A", weight: 4 },
    { node: "B", weight: 2 },
    { node: "D", weight: 1 },
    { node: "E", weight: 6 },
    { node: "F", weight: 3 },
  ],
  D: [
    { node: "B", weight: 5 },
    { node: "C", weight: 1 },
    { node: "F", weight: 2 },
  ],
  E: [
    { node: "B", weight: 8 },
    { node: "C", weight: 6 },
    { node: "F", weight: 4 },
    { node: "G", weight: 7 },
  ],
  F: [
    { node: "C", weight: 3 },
    { node: "D", weight: 2 },
    { node: "G", weight: 5 },
    { node: "H", weight: 1 },
  ],
  G: [
    { node: "E", weight: 7 },
    { node: "F", weight: 5 },
  ],
  H: [
    { node: "F", weight: 1 },
    { node: "I", weight: 3 },
  ],
  I: [{ node: "H", weight: 3 }],
};

const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");

const nodes = {
  A: { x: 100, y: 100 },
  B: { x: 300, y: 100 },
  C: { x: 200, y: 300 },
  D: { x: 400, y: 300 },
  E: { x: 500, y: 200 },
  F: { x: 300, y: 400 },
  G: { x: 400, y: 400 },
  H: { x: 200, y: 500 },
  I: { x: 100, y: 400 },
};

function drawNode(node, x, y) {
  ctx.beginPath();
  ctx.arc(x, y, 20, 0, 2 * Math.PI);
  ctx.fillStyle = "lightblue";
  ctx.fill();
  ctx.stroke();
  ctx.fillStyle = "black";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText(node, x, y);
}

function drawEdge(x1, y1, x2, y2, weight, color = "black") {
  ctx.beginPath();
  ctx.moveTo(x1, y1);
  ctx.lineTo(x2, y2);
  ctx.strokeStyle = color;
  ctx.stroke();

  // Calculate the midpoint for the text
  const midX = (x1 + x2) / 2;
  const midY = (y1 + y2) / 2;

  // Draw the weight text
  ctx.fillStyle = "red";
  ctx.fillText(weight, midX, midY);
}

function drawGraph(graph, shortestPathEdges = []) {
  // Draw edges first
  for (const node in graph) {
    const edges = graph[node];
    const { x: startX, y: startY } = nodes[node];
    edges.forEach(({ node: endNode, weight }) => {
      const { x: endX, y: endY } = nodes[endNode];
      const isShortestPathEdge = shortestPathEdges.some(
        (edge) =>
          (edge[0] === node && edge[1] === endNode) ||
          (edge[0] === endNode && edge[1] === node)
      );
      const color = isShortestPathEdge ? "green" : "black";
      drawEdge(startX, startY, endX, endY, weight, color);
    });
  }

  // Draw nodes on top
  for (const node in nodes) {
    const { x, y } = nodes[node];
    drawNode(node, x, y);
  }
}

function dijkstra(graph, startNode) {
  const distances = {};
  const previous = {};
  const visited = new Set();

  // Initialize distances and previous
  for (const node in graph) {
    distances[node] = Infinity;
    previous[node] = null;
  }
  distances[startNode] = 0;

  // Priority queue for the unvisited nodes
  const unvisitedNodes = new Set(Object.keys(graph));

  while (unvisitedNodes.size > 0) {
    // Find the unvisited node with the smallest distance
    let currentNode = null;
    unvisitedNodes.forEach((node) => {
      if (currentNode === null || distances[node] < distances[currentNode]) {
        currentNode = node;
      }
    });

    // If the smallest distance is infinity, break (disconnected graph)
    if (distances[currentNode] === Infinity) {
      break;
    }

    // Remove the current node from the unvisited set
    unvisitedNodes.delete(currentNode);

    // Update distances to neighboring nodes
    graph[currentNode].forEach(({ node: neighbor, weight }) => {
      if (!visited.has(neighbor)) {
        const newDist = distances[currentNode] + weight;
        if (newDist < distances[neighbor]) {
          distances[neighbor] = newDist;
          previous[neighbor] = currentNode;
        }
      }
    });

    // Mark the current node as visited
    visited.add(currentNode);
  }

  return { distances, previous };
}

function shortestPath(graph, startNode, endNode) {
  const { distances, previous } = dijkstra(graph, startNode);
  const path = [];
  let currentNode = endNode;

  while (currentNode !== null) {
    path.unshift(currentNode);
    currentNode = previous[currentNode];
  }

  if (distances[endNode] === Infinity) {
    return { path: null, distance: Infinity, edges: [] };
  }

  const edges = [];
  for (let i = 0; i < path.length - 1; i++) {
    edges.push([path[i], path[i + 1]]);
  }

  return { path, distance: distances[endNode], edges };
}

// Example usage
const startNode = "A";
const endNode = "D";
const result = shortestPath(graphData, startNode, endNode);

console.log(`Shortest path from ${startNode} to ${endNode}:`, result.path);
console.log(`Distance:`, result.distance);

// Draw the graph and highlight the shortest path edges
drawGraph(graphData, result.edges);
