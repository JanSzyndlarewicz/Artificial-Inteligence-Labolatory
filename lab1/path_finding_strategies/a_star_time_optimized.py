import heapq
from datetime import datetime

import networkx as nx

from lab1.heuristics import HeuristicFunction
from lab1.path_finding_strategies.a_star_time import AStarTimeStrategy
from lab1.trip_selection_strategies import BestTripSelectionStrategy


class AStarOptimizedTimeStrategy(AStarTimeStrategy):
    def __init__(self, best_trip_strategy: BestTripSelectionStrategy, heuristic_func: HeuristicFunction):
        super().__init__(best_trip_strategy, heuristic_func)

    def find_path(self, graph: nx.DiGraph, start: str, end: str, start_time: datetime) -> tuple[float, list[dict]]:
        self.initialize(graph, start, start_time)

        pq = [(0, start)]
        visited_nodes = 0
        best_known = {start: 0}

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
                if best_trip is None or (best_trip["departure_time"] - current_arrival).total_seconds() > 30 * 60:
                    continue

                wait_time = (best_trip["departure_time"] - current_arrival).total_seconds()
                new_cost = current_cost + wait_time + best_trip["duration"]

                if neighbor not in best_known or new_cost < best_known[neighbor]:
                    heuristic_adjustment = self.heuristic_func(neighbor, end, graph)
                    new_f = new_cost + heuristic_adjustment

                    best_known[neighbor] = new_cost
                    self.update_node(graph, pq, current, neighbor, best_trip, new_cost, new_f)

        self.logger.info(f"Visited nodes: {visited_nodes}")

        return float("inf"), []
