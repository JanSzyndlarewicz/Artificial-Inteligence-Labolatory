import heapq
from abc import ABC, abstractmethod
from datetime import datetime

import networkx as nx

from heuristics import HeuristicFunction
from trip_selection_strategies import BestTripSelectionStrategy


class PathfindingStrategy(ABC):

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
    ):
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
    def find_path(self, graph: nx.DiGraph, start: str, end: str, start_time: datetime):
        pass


class DijkstraTimeStrategy(PathfindingStrategy):
    def __init__(self, best_trip_strategy: BestTripSelectionStrategy):
        self.best_trip_strategy = best_trip_strategy

    def find_path(self, graph: nx.DiGraph, start: str, end: str, start_time: datetime) -> tuple[float, list[dict]]:

        self.initialize(graph, start, start_time)

        pq = [(0, start)]
        visited_nodes = 0

        while pq:
            cost, current = heapq.heappop(pq)
            if cost > graph.nodes[current]["cost"]:
                continue

            visited_nodes += 1
            current_arrival = graph.nodes[current]["arrival"]

            for root, neighbor, data in graph.edges(current, data=True):
                best_trip = self.best_trip_strategy.get_best_trip(
                    data["trips"],
                    current_arrival,
                    graph.nodes[root]["timetable"][-1].get("line") if graph.nodes[root]["timetable"] else None,
                )
                if best_trip is None:
                    continue

                wait_time = (best_trip["departure_time"] - current_arrival).total_seconds()
                new_cost = cost + wait_time + best_trip["duration"]

                if new_cost < graph.nodes[neighbor]["cost"]:
                    self.update_node(graph, pq, current, neighbor, best_trip, new_cost, new_cost)

        if graph.nodes[end]["timetable"]:
            return graph.nodes[end]["cost"], graph.nodes[end]["timetable"]
        else:
            return float("inf"), []


class DijkstraTransfersStrategy(PathfindingStrategy):
    def __init__(self, best_trip_strategy: BestTripSelectionStrategy):
        self.best_trip_strategy = best_trip_strategy

    def find_path(self, graph: nx.DiGraph, start: str, end: str, start_time: datetime) -> tuple[float, list[dict]]:

        self.initialize(graph, start, start_time)

        pq = [(0, start)]

        while pq:
            transfers, current = heapq.heappop(pq)
            if transfers > graph.nodes[current]["cost"]:
                continue

            current_arrival = graph.nodes[current]["arrival"]

            for root, neighbor, data in graph.edges(current, data=True):
                best_trip = self.best_trip_strategy.get_best_trip(
                    data["trips"],
                    current_arrival,
                    graph.nodes[root]["timetable"][-1].get("line") if graph.nodes[root]["timetable"] else None,
                )
                if best_trip is None:
                    continue

                new_cost = transfers + (
                    1
                    if not graph.nodes[root]["timetable"]
                    or graph.nodes[root]["timetable"][-1]["line"] != best_trip["line"]
                    else 0
                )

                if new_cost < graph.nodes[neighbor]["cost"]:
                    self.update_node(graph, pq, current, neighbor, best_trip, new_cost, new_cost)

        if graph.nodes[end]["timetable"]:
            return graph.nodes[end]["cost"], graph.nodes[end]["timetable"]
        else:
            return float("inf"), []


class AStarTimeStrategy(PathfindingStrategy):
    def __init__(self, best_trip_strategy: BestTripSelectionStrategy, heuristic_func: HeuristicFunction):
        self.best_trip_strategy = best_trip_strategy
        self.heuristic_func = heuristic_func

    def find_path(self, graph: nx.DiGraph, start: str, end: str, start_time: datetime) -> tuple[float, list[dict]]:
        self.initialize(graph, start, start_time)

        pq = [(0, start)]
        visited_nodes = 0

        while pq:
            current_f, current = heapq.heappop(pq)
            if current == end:
                return graph.nodes[end]["cost"], graph.nodes[end]["timetable"]

            visited_nodes += 1

            current_cost = graph.nodes[current]["cost"]
            current_arrival = graph.nodes[current]["arrival"]

            for root, neighbor, data in graph.edges(current, data=True):
                best_trip = self.best_trip_strategy.get_best_trip(
                    data["trips"],
                    current_arrival,
                    graph.nodes[root]["timetable"][-1].get("line") if graph.nodes[root]["timetable"] else None,
                )
                if best_trip is None:
                    continue

                wait_time = (best_trip["departure_time"] - current_arrival).total_seconds()
                new_cost = current_cost + wait_time + best_trip["duration"]

                new_f = new_cost + self.heuristic_func(neighbor, end, graph)

                if new_cost < graph.nodes[neighbor]["cost"]:
                    self.update_node(graph, pq, current, neighbor, best_trip, new_cost, new_f)

        return float("inf"), []


class AStarTransfersStrategy(PathfindingStrategy):
    def __init__(self, best_trip_strategy: BestTripSelectionStrategy, heuristic_func: HeuristicFunction):
        self.best_trip_strategy = best_trip_strategy
        self.heuristic_func = heuristic_func

    def find_path(self, graph: nx.DiGraph, start: str, end: str, start_time: datetime) -> tuple[float, list[dict]]:
        self.initialize(graph, start, start_time)

        pq = [(0, start)]
        visited_nodes = 0

        while pq:
            current_f, current = heapq.heappop(pq)
            if current == end:
                return graph.nodes[end]["cost"], graph.nodes[end]["timetable"]

            visited_nodes += 1

            current_cost = graph.nodes[current]["cost"]
            current_arrival = graph.nodes[current]["arrival"]

            for root, neighbor, data in graph.edges(current, data=True):
                best_trip = self.best_trip_strategy.get_best_trip(
                    data["trips"],
                    current_arrival,
                    graph.nodes[root]["timetable"][-1].get("line") if graph.nodes[root]["timetable"] else None,
                )
                if best_trip is None:
                    continue

                new_cost = current_cost + (
                    1
                    if not graph.nodes[root]["timetable"]
                    or graph.nodes[root]["timetable"][-1]["line"] != best_trip["line"]
                    else 0
                )

                new_f = new_cost + self.heuristic_func(neighbor, end, graph)

                if new_cost < graph.nodes[neighbor]["cost"]:
                    self.update_node(graph, pq, current, neighbor, best_trip, new_cost, new_f)

        return float("inf"), []
