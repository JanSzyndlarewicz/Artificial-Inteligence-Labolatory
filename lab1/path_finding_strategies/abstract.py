import heapq
import logging
from abc import ABC, abstractmethod
from datetime import datetime
from functools import lru_cache

import networkx as nx

from lab1.trip_selection_strategies import BestTripSelectionStrategy


class PathfindingStrategy(ABC):

    def __init__(self, best_trip_strategy: BestTripSelectionStrategy):
        self.logger = logging.getLogger(__name__)
        self.best_trip_strategy = best_trip_strategy

    @staticmethod
    def initialize(graph: nx.DiGraph, start: str, start_time: datetime):
        for node in graph.nodes:
            graph.nodes[node]["cost"] = float("inf")
            graph.nodes[node]["arrival"] = None
            graph.nodes[node]["timetable"] = []

        graph.nodes[start]["cost"] = 0
        graph.nodes[start]["arrival"] = start_time

    @staticmethod
    def update_node(
        graph: nx.DiGraph,
        pq: list[tuple[float, str]],
        current: str,
        neighbor: str,
        best_trip: dict,
        new_cost: float,
        new_f: float,
    ) -> None:
        graph.nodes[neighbor]["cost"] = new_cost
        graph.nodes[neighbor]["arrival"] = best_trip["arrival_time"]
        graph.nodes[neighbor]["timetable"] = graph.nodes[current].get("timetable", []) + [
            {
                "from": current,
                "to": neighbor,
                "departure_time": best_trip["departure_time"],
                "arrival_time": best_trip["arrival_time"],
                "line": best_trip["line"],
            }
        ]
        heapq.heappush(pq, (new_f, neighbor))

    @abstractmethod
    @lru_cache(1024)
    def find_path(self, graph: nx.DiGraph, start: str, end: str, start_time: datetime) -> tuple[float, list[dict]]:
        pass
