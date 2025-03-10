import heapq
from abc import ABC, abstractmethod
from datetime import datetime
from math import sqrt
from typing import Any

import networkx as nx


class PathfindingStrategy(ABC):
    @abstractmethod
    def find_path(self, graph: nx.DiGraph, start: str, end: str, start_time: datetime, heuristic_func=None):
        pass


class DijkstraTimeStrategy(PathfindingStrategy):
    def find_path(
        self, graph: nx.DiGraph, start: str, end: str, start_time: datetime, heuristic_func=None
    ) -> tuple[float, list[dict]]:

        for node in graph.nodes:
            graph.nodes[node]["cost"] = float("inf")
            graph.nodes[node]["arrival"] = None
            graph.nodes[node]["timetable"] = []

        graph.nodes[start]["cost"] = 0
        graph.nodes[start]["arrival"] = start_time

        pq = [(0, start)]

        while pq:
            cost, current = heapq.heappop(pq)
            if cost > graph.nodes[current]["cost"]:
                continue

            current_arrival = graph.nodes[current]["arrival"]
            for root, neighbor, data in graph.edges(current, data=True):

                best_trip = self.get_best_trip(
                    data["trips"],
                    current_arrival,
                    graph.nodes[root]["timetable"][-1].get("line") if graph.nodes[root]["timetable"] else None,
                )
                if best_trip is None:
                    continue

                wait_time = (best_trip["departure_time"] - current_arrival).total_seconds()

                new_cost = cost + wait_time + best_trip["duration"]

                if new_cost < graph.nodes[neighbor]["cost"]:
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
                    heapq.heappush(pq, (new_cost, neighbor))

        if graph.nodes[end]["timetable"]:
            return graph.nodes[end]["cost"], graph.nodes[end]["timetable"]
        else:
            return float("inf"), []

    def get_best_trip(self, trips: list[dict], arrival_time: datetime, current_line: str | None) -> dict | None:
        best_trip = None
        for trip in trips:
            if trip["departure_time"] >= arrival_time:
                if (
                    best_trip is None
                    or trip["departure_time"] < best_trip["departure_time"]
                    or (trip["departure_time"] == best_trip["departure_time"] and trip["line"] == current_line)
                ):
                    best_trip = trip
        return best_trip


class DijkstraTransfersStrategy(PathfindingStrategy):
    def find_path(
            self, graph: nx.DiGraph, start: str, end: str, start_time: datetime, heuristic_func=None
    ) -> tuple[Any, Any] | tuple[float, list[Any]]:
        for node in graph.nodes:
            graph.nodes[node]["cost"] = float("inf")
            graph.nodes[node]["arrival"] = None
            graph.nodes[node]["timetable"] = []

        graph.nodes[start]["cost"] = 0  # Start with 0 transfers
        graph.nodes[start]["arrival"] = start_time

        # Priority queue that prioritizes nodes with fewer transfers
        pq = [(0, start)]

        while pq:
            transfers, current = heapq.heappop(pq)
            if transfers > graph.nodes[current]["cost"]:
                continue

            current_arrival = graph.nodes[current]["arrival"]

            for root, neighbor, data in graph.edges(current, data=True):
                best_trip = self.get_best_trip(
                    data["trips"],
                    current_arrival,
                    graph.nodes[root]["timetable"][-1].get("line") if graph.nodes[root]["timetable"] else None,
                )
                if best_trip is None:
                    continue

                # Track the number of transfers
                new_transfers = transfers + (
                    1 if not graph.nodes[root]["timetable"] or graph.nodes[root]["timetable"][-1]["line"] != best_trip[
                        "line"]
                    else 0
                )

                # If this path has fewer transfers, update the node
                if new_transfers < graph.nodes[neighbor]["cost"]:
                    graph.nodes[neighbor]["cost"] = new_transfers
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
                    heapq.heappush(pq, (new_transfers, neighbor))

        if graph.nodes[end]["timetable"]:
            return graph.nodes[end]["cost"], graph.nodes[end]["timetable"]
        else:
            return float("inf"), []

    def get_best_trip(self, trips: list[dict], arrival_time: datetime, current_line: str | None) -> dict | None:
        best_trip = None
        for trip in trips:
            if trip["departure_time"] >= arrival_time:
                if current_line is None or trip["line"] == current_line:
                    if best_trip is None or trip["departure_time"] < best_trip["departure_time"]:
                        best_trip = trip
                elif best_trip is None or trip["departure_time"] < best_trip["departure_time"]:
                    best_trip = trip
        return best_trip


def heuristic(neighbor, end, graph):
    x1, y1 = float(graph.nodes[neighbor]["x"]), float(graph.nodes[neighbor]["y"])
    x2, y2 = float(graph.nodes[end]["x"]), float(graph.nodes[end]["y"])
    return sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


class AStarTimeStrategy(PathfindingStrategy):
    def find_path(self, graph: nx.DiGraph, start: str, end: str, start_time: datetime, heuristic_func: callable=None):
        for node in graph.nodes:
            graph.nodes[node]["cost"] = float("inf")
            graph.nodes[node]["arrival"] = None
            graph.nodes[node]["timetable"] = []

        graph.nodes[start]["cost"] = 0
        graph.nodes[start]["arrival"] = start_time

        pq = [(0, start)]

        while pq:
            current_f, current = heapq.heappop(pq)
            if current == end:
                break  # Found the destination

            current_cost = graph.nodes[current]["cost"]
            current_arrival = graph.nodes[current]["arrival"]

            for root, neighbor, data in graph.edges(current, data=True):
                best_trip = self.get_best_trip(
                    data["trips"],
                    current_arrival,
                    graph.nodes[root]["timetable"][-1].get("line") if graph.nodes[root]["timetable"] else None,
                )
                if best_trip is None:
                    continue

                wait_time = (best_trip["departure_time"] - current_arrival).total_seconds()
                new_cost = current_cost + wait_time + best_trip["duration"]

                # Heuristic function for estimating the time to the goal (using coordinates)
                h = heuristic(neighbor, end, graph)  # Heuristic function should estimate time to the end node

                new_f = new_cost + h

                if new_cost < graph.nodes[neighbor]["cost"]:
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

        if graph.nodes[end]["timetable"]:
            return graph.nodes[end]["cost"], graph.nodes[end]["timetable"]
        else:
            return float("inf"), []

    def get_best_trip(self, trips: list[dict], arrival_time: datetime, current_line: str | None) -> dict | None:
        best_trip = None
        for trip in trips:
            if trip["departure_time"] >= arrival_time:
                if (
                    best_trip is None
                    or trip["departure_time"] < best_trip["departure_time"]
                    or (trip["departure_time"] == best_trip["departure_time"] and trip["line"] == current_line)
                ):
                    best_trip = trip
        return best_trip


class AStarTransfersStrategy(PathfindingStrategy):
    def find_path(self, graph: nx.DiGraph, start: str, end: str, start_time: datetime, heuristic_func: callable=None):
        for node in graph.nodes:
            graph.nodes[node]["cost"] = float("inf")
            graph.nodes[node]["arrival"] = None
            graph.nodes[node]["timetable"] = []

        graph.nodes[start]["cost"] = 0
        graph.nodes[start]["arrival"] = start_time

        pq = [(0, start)]  # Priority queue stores (f(n), node)

        while pq:
            current_f, current = heapq.heappop(pq)
            if current == end:
                break  # Found the destination

            current_cost = graph.nodes[current]["cost"]
            current_arrival = graph.nodes[current]["arrival"]

            for root, neighbor, data in graph.edges(current, data=True):
                best_trip = self.get_best_trip(
                    data["trips"],
                    current_arrival,
                    graph.nodes[root]["timetable"][-1].get("line") if graph.nodes[root]["timetable"] else None,
                )
                if best_trip is None:
                    continue

                # Track the number of transfers
                new_transfers = current_cost + (
                    1 if not graph.nodes[root]["timetable"] or graph.nodes[root]["timetable"][-1]["line"] != best_trip["line"]
                    else 0
                )

                # Heuristic function for estimating the transfers to the goal
                h = heuristic(neighbor, end)  # Heuristic function should estimate transfers to the end node

                new_f = new_transfers + h

                if new_transfers < graph.nodes[neighbor]["cost"]:
                    graph.nodes[neighbor]["cost"] = new_transfers
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

        if graph.nodes[end]["timetable"]:
            return graph.nodes[end]["cost"], graph.nodes[end]["timetable"]
        else:
            return float("inf"), []

    def get_best_trip(self, trips: list[dict], arrival_time: datetime, current_line: str | None) -> dict | None:
        best_trip = None
        for trip in trips:
            if trip["departure_time"] >= arrival_time:
                if current_line is None or trip["line"] == current_line:
                    if best_trip is None or trip["departure_time"] < best_trip["departure_time"]:
                        best_trip = trip
                elif best_trip is None or trip["departure_time"] < best_trip["departure_time"]:
                    best_trip = trip
        return best_trip
