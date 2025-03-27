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

    def compute_cost(
        self, graph: nx.DiGraph, route: list[str], start_time: datetime
    ) -> tuple[float, list[Any], list[float]]:
        total_cost = 0
        full_path = []
        prev_line = None
        segment_costs = []

        for i in range(len(route) - 1):
            u, v = route[i], route[i + 1]
            cost, path = self.astar_strategy.find_path(graph, u, v, start_time)
            if cost == float("inf"):
                return float("inf"), [], []

            total_cost += cost
            full_path.extend(path)
            segment_costs.append(cost)

            if (
                prev_line
                and path
                and prev_line != path[0]["line"]
                and isinstance(self.astar_strategy.best_trip_strategy, TransferBasedBestTripSelection)
            ):
                total_cost -= 1

            prev_line = path[-1]["line"] if path else None
            start_time = path[-1]["arrival_time"] if path else start_time

            self.logger.debug(f"Step cost: {cost}, route: {route}, time: {start_time}")

        return total_cost, full_path, segment_costs

    def tabu_search(
            self, graph: nx.DiGraph, start: str, stops: list[str], start_time: datetime, max_iter: int = 100
    ) -> tuple[list[str], list[dict], float]:
        best_route, best_cost = [start] + stops + [start], float("inf")
        best_path = []
        tabu_size = max(5, len(stops))
        tabu_edges = set()

        self.logger.info(f"Starting tabu search: start={start}, stops={stops}, time={start_time}")

        for _ in range(max_iter):
            total_cost, full_path, segment_costs = self.compute_cost(graph, best_route, start_time)

            if total_cost < best_cost:
                best_cost = total_cost
                best_path = full_path

            if not segment_costs:
                break

            # Generate weighted choices for swapping stops to prioritize the most expensive segments
            weighted_choices = [i for i, cost in enumerate(segment_costs[1:-1]) for _ in range(int(cost * 10))]
            if not weighted_choices:
                continue

            # Retry if trying to swap the same stops
            i, j = 1, 1
            attempts = 0
            while attempts < 10:
                i, j = sorted(random.sample(weighted_choices, 2))
                if i != j:
                    break
                attempts += 1

            new_route = best_route[:]
            new_route[i + 1], new_route[j + 1] = new_route[j + 1], new_route[i + 1]

            # Check if new route contains forbidden edges
            new_edges = {(new_route[k], new_route[k + 1]) for k in range(len(new_route) - 1)}
            if any(edge in tabu_edges for edge in new_edges):
                continue

            total_cost, full_path, _ = self.compute_cost(graph, new_route, start_time)

            if total_cost < best_cost:
                best_route, best_cost, best_path = new_route, total_cost, full_path

            # Add new edge to tabu list
            if len(new_edges) > 0:
                tabu_edges.add(random.choice(list(new_edges)))
            if len(tabu_edges) > tabu_size:
                tabu_edges.pop()  # Delete oldest edge

            self.logger.debug(f"Evaluated route: {new_route}, cost: {total_cost}")

        self.logger.info(f"Best found route: {best_route}, cost: {best_cost}")

        return best_route, best_path, best_cost

