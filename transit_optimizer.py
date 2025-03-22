import logging
import random
from datetime import datetime

import networkx as nx

from path_finding_strategies import PathfindingStrategy
from trip_selection_strategies import TransferBasedBestTripSelection


class TransitOptimizer:
    def __init__(self, path_finder: PathfindingStrategy):
        self.logger = logging.getLogger(__name__)
        self.astar_strategy = path_finder

    def compute_cost(self, graph: nx.DiGraph, route: list[str], start_time: datetime) -> float | tuple[int, list[dict]]:
        total_cost = 0
        full_path = []
        prev_line = None

        for i in range(len(route) - 1):
            u, v = route[i], route[i + 1]
            cost, path = self.astar_strategy.find_path(graph, u, v, start_time)
            if cost == float("inf"):
                return float("inf")

            full_path += path

            self.logger.debug(f"Computed cost: {cost} for route: {route}, start time: {start_time}")

            total_cost += cost
            if (
                path
                and prev_line
                and prev_line != path[0]["line"]
                and isinstance(self.astar_strategy.best_trip_strategy, TransferBasedBestTripSelection)
            ):
                total_cost += -1
            prev_line = path[-1]["line"] if path else None

        return total_cost, full_path

    def tabu_search(
        self,
        graph: nx.DiGraph,
        start: str,
        stops: list[str],
        start_time: datetime,
        tabu_size: int = 10,
        max_iter: int = 100,
    ) -> tuple[list[str], float]:
        best_route, best_cost = [start] + stops + [start], float("inf")
        best_path = []
        tabu_list = []

        self.logger.info(f"Starting tabu search with start: {start}, stops: {stops}, start time: {start_time}")

        for _ in range(max_iter):
            neighbors = [list(best_route) for _ in range(len(stops))]
            for n in neighbors:
                i, j = sorted((random.randint(1, len(stops)), random.randint(1, len(stops))))
                n[i], n[j] = n[j], n[i]
                if n in tabu_list:
                    continue

                current_start_time = start_time
                total_cost = 0
                full_path = []

                for k in range(len(n) - 1):
                    u, v = n[k], n[k + 1]
                    cost, path = self.compute_cost(graph, [u, v], current_start_time)
                    if cost == float("inf"):
                        total_cost = float("inf")
                        break

                    total_cost += cost
                    full_path += path
                    current_start_time = path[-1]["arrival_time"] if path else current_start_time

                self.logger.debug(
                    f"Computed cost: {total_cost} for route: {n}, tabu list: {tabu_list}, start time: {start_time}"
                )
                if total_cost < best_cost:
                    best_route, best_cost = n, total_cost
                    tabu_list.append(n)
                    if len(tabu_list) > tabu_size:
                        tabu_list.pop(0)
                    best_path = full_path

        self.logger.info(f"Best route: {best_route}, best cost: {best_cost}")

        for trip in best_path:
            print(f"{trip['line']} {trip['departure_time']} {trip['from']} -> {trip['arrival_time']} {trip['to']}")

        return best_route, best_cost
