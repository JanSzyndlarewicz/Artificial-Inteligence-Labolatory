import heapq
from datetime import datetime

import networkx as nx

from path_finding_strategies.abstract import PathfindingStrategy
from trip_selection_strategies import BestTripSelectionStrategy


class DijkstraTransfersStrategy(PathfindingStrategy):
    def __init__(self, best_trip_strategy: BestTripSelectionStrategy):
        super().__init__(best_trip_strategy)

    def find_path(self, graph: nx.DiGraph, start: str, end: str, start_time: datetime) -> tuple[float, list[dict]]:

        self.initialize(graph, start, start_time)

        pq = [(0, start)]
        visited_nodes = 0

        while pq:
            transfers, current = heapq.heappop(pq)
            if transfers > graph.nodes[current]["cost"]:
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

                new_cost = transfers + (
                    1
                    if not graph.nodes[root]["timetable"]
                    or graph.nodes[root]["timetable"][-1]["line"] != best_trip["line"]
                    else 0
                )

                if new_cost < graph.nodes[neighbor]["cost"]:
                    self.update_node(graph, pq, current, neighbor, best_trip, new_cost, new_cost)

        self.logger.info(f"Visited nodes: {visited_nodes}")

        if graph.nodes[end]["timetable"]:
            return graph.nodes[end]["cost"], graph.nodes[end]["timetable"]
        else:
            return float("inf"), []
