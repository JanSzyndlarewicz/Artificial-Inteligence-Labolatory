import unittest
from datetime import datetime

import networkx as nx

from path_finder import DijkstraTransfersStrategy


class TestDijkstraTransfersStrategy(unittest.TestCase):
    def setUp(self):
        self.graph = nx.DiGraph()
        self.graph.add_nodes_from(["A", "B", "C", "D"])
        self.graph.add_edges_from([
            ("A", "B", {"trips": [{"departure_time": datetime(2025, 3, 10, 8, 0), "arrival_time": datetime(2025, 3, 10, 8, 15), "line": "1", "duration": 900}]}),
            ("B", "C", {"trips": [{"departure_time": datetime(2025, 3, 10, 8, 30), "arrival_time": datetime(2025, 3, 10, 8, 45), "line": "2", "duration": 900}]}),  # Different line
            ("C", "D", {"trips": [{"departure_time": datetime(2025, 3, 10, 9, 0), "arrival_time": datetime(2025, 3, 10, 9, 20), "line": "2", "duration": 1200}]}),
        ])
        self.strategy = DijkstraTransfersStrategy()

    def test_find_path_with_fewest_transfers(self):
        start_time = datetime(2025, 3, 10, 7, 50)
        cost, timetable = self.strategy.find_path(self.graph, "A", "D", start_time)
        expected_cost = 2  # One transfer (A -> B -> C -> D)
        self.assertEqual(cost, expected_cost)
        self.assertEqual(len(timetable), 3)  # 3 trips (A -> B, B -> C, C -> D)

    def test_find_path_without_transfer(self):
        # Add an edge that goes directly from A to D with no transfer
        self.graph.add_edge("A", "D", trips=[{
            "departure_time": datetime(2025, 3, 10, 8, 0),
            "arrival_time": datetime(2025, 3, 10, 8, 30),
            "line": "1",
            "duration": 1800,
        }])
        start_time = datetime(2025, 3, 10, 7, 50)
        cost, timetable = self.strategy.find_path(self.graph, "A", "D", start_time)
        expected_cost = 1  # No transfer (direct path)
        self.assertEqual(cost, expected_cost)
        self.assertEqual(len(timetable), 1)  # One trip (A -> D)

    def test_no_path(self):
        # Add no connection between nodes A and D
        self.graph.remove_edge("C", "D")
        start_time = datetime(2025, 3, 10, 7, 50)
        cost, timetable = self.strategy.find_path(self.graph, "A", "D", start_time)
        self.assertEqual(cost, float("inf"))
        self.assertEqual(timetable, [])
