import copy
import json
import logging
import os
import pickle
from bisect import insort
from datetime import datetime, timedelta

import networkx as nx
import pandas as pd
from pandas import Timestamp

from config import GRAPH_FILE_PATH
from path_finding_strategies import PathfindingStrategy
from transit_optimizer import TransitOptimizer


class TransitGraph:
    def __init__(self, file_path: str) -> None:
        self.logger = logging.getLogger(__name__)
        self.df = None
        self.file_path = file_path
        self.graph = nx.DiGraph()
        self.build_graph()

    def load_data(self, file_path: str) -> None:
        try:
            self.logger.info(f"Loading data from {file_path}")
            self.df = pd.read_csv(file_path, dtype=str)

            self.df.dropna(inplace=True)

            time_cols = ["departure_time", "arrival_time"]
            for col in time_cols:
                self.df[col] = self.df[col].apply(self.clean_time).astype(str)
                self.df[col] = pd.to_datetime(self.df[col], format="%H:%M:%S").dt.time

            self.df["departure_time"] = pd.to_datetime(self.df["departure_time"], format="%H:%M:%S").dt.time
            self.df["arrival_time"] = pd.to_datetime(self.df["arrival_time"], format="%H:%M:%S").dt.time

            # Create base datetime (to ensure calculations work properly)
            base_date = datetime(1900, 1, 2)

            # Convert to full datetime
            self.df["departure_datetime"] = self.df["departure_time"].apply(lambda x: datetime.combine(base_date, x))
            self.df["arrival_datetime"] = self.df["arrival_time"].apply(lambda x: datetime.combine(base_date, x))

            # Handle cases where arrival time is on the next day
            self.df.loc[self.df["arrival_time"] < self.df["departure_time"], "departure_datetime"] -= timedelta(days=1)

            # Assign the final values
            self.df["departure_time"] = self.df["departure_datetime"]
            self.df["arrival_time"] = self.df["arrival_datetime"]

            self.df.drop(columns=["departure_datetime", "arrival_datetime"], inplace=True)

            self.df["departure_time"] = pd.to_datetime(self.df["departure_time"])
            self.df["arrival_time"] = pd.to_datetime(self.df["arrival_time"])

            self.df["duration"] = (self.df["arrival_time"] - self.df["departure_time"]).dt.total_seconds()

            self.df["start_stop"] = self.df["start_stop"].str.title()
            self.df["end_stop"] = self.df["end_stop"].str.title()

        except Exception as e:
            raise ValueError(f"Error loading data: {e}")

    @staticmethod
    def clean_time(time_str: str) -> str:
        try:
            hours, minutes, seconds = map(int, time_str.split(":"))
            return f"{hours % 24:02}:{minutes:02}:{seconds:02}"
        except ValueError:
            raise ValueError(f"Invalid time format: {time_str}")

    def build_graph(self) -> None:
        if os.path.exists(GRAPH_FILE_PATH):
            self.logger.info(f"Loading graph from {GRAPH_FILE_PATH}")
            self.load_graph(GRAPH_FILE_PATH, format="graphml")
        else:
            self.logger.info("Loading data for new graph")
            self.load_data(self.file_path)

            self.logger.info("Building a graph")
            for _, row in self.df.iterrows():
                start, end = row["start_stop"], row["end_stop"]

                self.graph.add_node(start, x=row["start_stop_lat"], y=row["start_stop_lon"])
                self.graph.add_node(end, x=row["end_stop_lat"], y=row["end_stop_lon"])

                self.graph.add_edge(start, end)
                trips = self.graph[start][end].setdefault("trips", [])

                trip = {
                    "departure_time": row["departure_time"],
                    "arrival_time": row["arrival_time"],
                    "duration": row["duration"],
                    "line": row["line"],
                }

                insort(trips, trip, key=lambda t: t["arrival_time"])

            self.save_graph(GRAPH_FILE_PATH, format="graphml")

    def load_graph(self, filename: str, format: str = "graphml") -> None:
        if format == "graphml":
            self.graph = nx.read_graphml(filename)

            for u, v, data in self.graph.edges(data=True):
                if "trips" in data:
                    data["trips"] = json.loads(data["trips"])
                    for trip in data["trips"]:
                        trip["departure_time"] = pd.Timestamp(trip["departure_time"])
                        trip["arrival_time"] = pd.Timestamp(trip["arrival_time"])
        else:
            raise ValueError("Unsupported format. Use 'graphml', 'gml', 'json', or 'pickle'.")

    def save_graph(self, filename: str, format: str = "graphml") -> None:
        if format == "graphml":
            graph_copy = copy.deepcopy(self.graph)
            for u, v, data in graph_copy.edges(data=True):
                if "trips" in data:
                    for trip in data["trips"]:
                        if isinstance(trip["departure_time"], Timestamp):
                            trip["departure_time"] = trip["departure_time"].isoformat()
                        if isinstance(trip["arrival_time"], Timestamp):
                            trip["arrival_time"] = trip["arrival_time"].isoformat()

                    data["trips"] = json.dumps(data["trips"])

            nx.write_graphml(graph_copy, filename)
        else:
            raise ValueError("Unsupported format. Use 'graphml', 'gml', 'json', or 'pickle'.")

    def find_shortest_path(
        self, strategy: PathfindingStrategy, start: str, end: str, start_time_at_stop: datetime
    ) -> tuple[float, list[dict]]:
        cost, path = strategy.find_path(self.graph, start, end, start_time_at_stop)
        return cost, path

    def optimize_transit(
        self,
        transit_optimizer: TransitOptimizer,
        start: str,
        stops: list[str],
        start_time: datetime,
        max_iter: int = 100,
    ) -> tuple[list[str], float]:

        return transit_optimizer.tabu_search(self.graph, start, stops, start_time, max_iter)
