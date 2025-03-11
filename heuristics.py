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