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
        lat1, lon1 = math.radians(float(graph.nodes[neighbor]["x"])), math.radians(float(graph.nodes[neighbor]["y"]))
        lat2, lon2 = math.radians(float(graph.nodes[end]["x"])), math.radians(float(graph.nodes[end]["y"]))

        dlat = lat2 - lat1
        dlon = lon2 - lon1

        a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

        R = 6371
        distance = R * c
        return distance
