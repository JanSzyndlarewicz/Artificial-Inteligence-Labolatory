import unittest
from datetime import datetime, timedelta
from heapq import heappop, heappush

import networkx as nx

from path_finder import DijkstraTimeStrategy


class TestDijkstraTimeStrategy(unittest.TestCase):
    def setUp(self):
        self.graph = nx.DiGraph()
        self.graph.add_nodes_from(["A", "B", "C", "D"])
        self.graph.add_edges_from(
            [
                (
                    "A",
                    "B",
                    {
                        "trips": [
                            {
                                "departure_time": datetime(2025, 3, 10, 8, 0),
                                "arrival_time": datetime(2025, 3, 10, 8, 15),
                                "line": "1",
                                "duration": 900,
                            }
                        ]
                    },
                ),
                (
                    "B",
                    "C",
                    {
                        "trips": [
                            {
                                "departure_time": datetime(2025, 3, 10, 8, 30),
                                "arrival_time": datetime(2025, 3, 10, 8, 45),
                                "line": "1",
                                "duration": 900,
                            }
                        ]
                    },
                ),
                (
                    "C",
                    "D",
                    {
                        "trips": [
                            {
                                "departure_time": datetime(2025, 3, 10, 9, 0),
                                "arrival_time": datetime(2025, 3, 10, 9, 20),
                                "line": "1",
                                "duration": 1200,
                            }
                        ]
                    },
                ),
            ]
        )
        self.strategy = DijkstraTimeStrategy()

    def test_find_path(self):
        start_time = datetime(2025, 3, 10, 7, 50)
        cost, timetable = self.strategy.find_path(self.graph, "A", "D", start_time)
        expected_cost = 5400
        self.assertEqual(cost, expected_cost)
        self.assertEqual(len(timetable), 3)

    def test_no_path(self):
        # Add no connection between nodes A and D
        self.graph.remove_edge("C", "D")
        start_time = datetime(2025, 3, 10, 7, 50)
        cost, timetable = self.strategy.find_path(self.graph, "A", "D", start_time)
        self.assertEqual(cost, float("inf"))
        self.assertEqual(timetable, [])
