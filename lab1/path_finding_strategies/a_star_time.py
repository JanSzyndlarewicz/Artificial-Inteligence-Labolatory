import heapq
from datetime import datetime

import networkx as nx

from heuristics import HeuristicFunction
from path_finding_strategies.abstract import PathfindingStrategy
from trip_selection_strategies import BestTripSelectionStrategy


class AStarTimeStrategy(PathfindingStrategy):
    def __init__(self, best_trip_strategy: BestTripSelectionStrategy, heuristic_func: HeuristicFunction):
        super().__init__(best_trip_strategy)
        self.heuristic_func = heuristic_func

    def find_path(self, graph: nx.DiGraph, start: str, end: str, start_time: datetime) -> tuple[float, list[dict]]:
        self.initialize(graph, start, start_time)

        pq = [(0, start)]
        visited_nodes = 0

        while pq:
            current_f, current = heapq.heappop(pq)
            if current == end:
                self.logger.info(f"Visited nodes: {visited_nodes}")
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

        self.logger.info(f"Visited nodes: {visited_nodes}")

        return float("inf"), []
