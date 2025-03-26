import heapq
import logging
import random
from datetime import datetime
from typing import Any

import networkx as nx

from path_finding_strategies import PathfindingStrategy
from trip_selection_strategies import TransferBasedBestTripSelection


class TransitOptimizer:
    def __init__(self, path_finder: PathfindingStrategy):
        self.logger = logging.getLogger(__name__)
        self.astar_strategy = path_finder

    def compute_cost(self, graph: nx.DiGraph, route: list[str], start_time: datetime) -> tuple[float, list[Any]]:
        total_cost = 0
        full_path = []
        prev_line = None

        for i in range(len(route) - 1):
            u, v = route[i], route[i + 1]
            cost, path = self.astar_strategy.find_path(graph, u, v, start_time)
            if cost == float("inf"):
                return float("inf"), []

            total_cost += cost
            full_path.extend(path)

            if (
                prev_line and path and prev_line != path[0]["line"]
                and isinstance(self.astar_strategy.best_trip_strategy, TransferBasedBestTripSelection)
            ):
                total_cost += 1

            prev_line = path[-1]["line"] if path else None
            start_time = path[-1]["arrival_time"] if path else start_time

            self.logger.debug(f"Step cost: {cost}, route: {route}, time: {start_time}")

        return total_cost, full_path

    def tabu_search(
            self, graph: nx.DiGraph, start: str, stops: list[str], start_time: datetime, max_iter: int = 100,
            tabu_size: int = 10
    ) -> tuple[list[str], float]:
        best_route, best_cost = [start] + stops + [start], float("inf")
        best_path = []
        tabu_list = []

        self.logger.info(f"Starting tabu search: start={start}, stops={stops}, time={start_time}")

        for _ in range(max_iter):
            i, j = random.sample(range(1, len(stops) + 1), 2)
            new_route = best_route[:]
            new_route[i], new_route[j] = new_route[j], new_route[i]

            if new_route in tabu_list:
                continue

            total_cost, full_path = self.compute_cost(graph, new_route, start_time)

            if total_cost < best_cost:
                best_route, best_cost, best_path = new_route, total_cost, full_path
                heapq.heappush(tabu_list, (total_cost, new_route))
                if len(tabu_list) > tabu_size:
                    heapq.heappop(tabu_list)

            self.logger.debug(f"Evaluated route: {new_route}, cost: {total_cost}")

        self.logger.info(f"Best found route: {best_route}, cost: {best_cost}")

        for trip in best_path:
            print(f"{trip['line']} {trip['departure_time']} {trip['from']} -> {trip['arrival_time']} {trip['to']}")

        return best_route, best_cost
