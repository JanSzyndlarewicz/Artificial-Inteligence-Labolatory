import heapq
from abc import ABC, abstractmethod
from datetime import datetime


class PathfindingStrategy(ABC):
    @abstractmethod
    def find_path(self, graph, start, end, start_time_at_stop, heuristic=None):
        pass


class DijkstraTimeStrategy(PathfindingStrategy):
    def find_path(self, graph, start, end, start_time_at_stop, heuristic=None):
        pq = [(0, start)]
        visited = set()

        while pq:
            (cost, node) = heapq.heappop(pq)

            if node in visited:
                continue

            visited.add(node)

            for root, neighbor, data in graph.edges(node, data=True):
                if neighbor in visited:
                    continue

                arrival_time = self.get_arrival_time(graph.nodes[node]['path']) if graph.nodes[node].get('path') else start_time_at_stop
                trip = self.get_best_path(data['trips'], arrival_time)
                if trip is not None:
                    arrival_time_dt = datetime.strptime(arrival_time, '%H:%M:%S')
                    departure_time_dt = datetime.strptime(trip['departure_time'], '%H:%M:%S')
                    time_difference = (departure_time_dt - arrival_time_dt).total_seconds()

                    total_cost = cost + trip['duration'] + time_difference

                    if total_cost < graph.nodes[neighbor].get('cost', float("inf")):
                        graph.nodes[neighbor]['cost'] = total_cost
                        graph.nodes[neighbor]['path'] = [node, graph.nodes[node], trip]
                        heapq.heappush(pq, (total_cost, neighbor))

        if graph.nodes[end].get('path') is not None:
            return graph.nodes[end]
        else:
            return float("inf"), []

    def get_best_path(self, edges: list[dict], arrival_time: str) -> dict | None:
        best_edge = None
        for edge in edges:
            if edge['departure_time'] >= arrival_time:
                if best_edge is None:
                    best_edge = edge
                elif edge['departure_time'] < best_edge['departure_time']:
                    best_edge = edge
        return best_edge

    def get_arrival_time(self, path):
        return path[2]['arrival_time']


class AStarTimeStrategy(PathfindingStrategy):
    def find_path(self, graph, start, end, start_time_at_stop, heuristic):
        pq = [(0, start, [])]
        visited = set()
        while pq:
            (cost, node, path) = heapq.heappop(pq)
            if node in visited:
                continue
            visited.add(node)
            path = path + [node]

            if node == end:
                return cost, path

            for neighbor, _, data in graph.edges(node, data=True):
                if neighbor not in visited:
                    total_cost = cost + data['duration']
                    estimated_cost = total_cost + heuristic(neighbor, end)
                    heapq.heappush(pq, (estimated_cost, neighbor, path))

        return float("inf"), []


class AStarTransfersStrategy(PathfindingStrategy):
    def find_path(self, graph, start, end, start_time_at_stop, heuristic):
        pq = [(0, start, [], None)]
        visited = set()
        while pq:
            (cost, node, path, last_line) = heapq.heappop(pq)
            if node in visited:
                continue
            visited.add(node)
            path = path + [node]

            if node == end:
                return cost, path

            for neighbor, key, data in graph.edges(node, data=True):
                if neighbor not in visited:
                    line = data['line']
                    # Increment transfer cost if line changes
                    transfer_cost = cost + (1 if line != last_line else 0)
                    heapq.heappush(pq, (transfer_cost, neighbor, path, line))

        return float("inf"), []
