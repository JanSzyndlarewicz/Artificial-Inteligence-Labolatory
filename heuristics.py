import math
from abc import ABC, abstractmethod
from math import sqrt
from typing import Any

import networkx as nx


class HeuristicFunction(ABC):
    @abstractmethod
    def __call__(self, neighbor: str, end: str, graph: nx.DiGraph) -> Any:
        pass


class EuclideanDistanceHeuristic(HeuristicFunction):
    def __call__(self, neighbor: str, end: str, graph: nx.DiGraph) -> Any:
        x1, y1 = float(graph.nodes[neighbor]["x"]), float(graph.nodes[neighbor]["y"])
        x2, y2 = float(graph.nodes[end]["x"]), float(graph.nodes[end]["y"])
        return sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


class HaversineDistanceHeuristic(HeuristicFunction):
    def __call__(self, neighbor: str, end: str, graph: nx.DiGraph) -> Any:
        return self.haversine_distance(graph.nodes[neighbor], graph.nodes[end])

    @staticmethod
    def haversine_distance(node1, node2):
        lat1, lon1 = math.radians(float(node1["x"])), math.radians(float(node1["y"]))
        lat2, lon2 = math.radians(float(node2["x"])), math.radians(float(node2["y"]))

        dlat = lat2 - lat1
        dlon = lon2 - lon1

        a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

        R = 6371
        return R * c


class HaversineMaxTimeHeuristic(HaversineDistanceHeuristic):
    def __call__(self, neighbor: str, end: str, graph: nx.DiGraph) -> Any:
        distance = self.haversine_distance(graph.nodes[neighbor], graph.nodes[end])

        speed = 74  # km/h
        return (distance / speed) * 3600

class HaversineAverageTimeHeuristic(HaversineDistanceHeuristic):
    def __call__(self, neighbor: str, end: str, graph: nx.DiGraph) -> Any:
        distance = self.haversine_distance(graph.nodes[neighbor], graph.nodes[end])

        speed = 25  # km/h
        return (distance / speed) * 3600

class HaversineCoefficientTransferHeuristic(HaversineDistanceHeuristic):
    def __call__(self, neighbor: str, end: str, graph: nx.DiGraph) -> Any:
        distance = self.haversine_distance(graph.nodes[neighbor], graph.nodes[end])
        coefficient = 0.5
        return distance * coefficient