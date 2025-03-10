from bisect import insort
from datetime import datetime, timedelta

import networkx as nx
import pandas as pd

from path_finder import PathfindingStrategy


class TransitGraph:
    def __init__(self, file_path: str, strategy: PathfindingStrategy) -> None:
        self.df = None
        self.graph = nx.DiGraph()
        self.strategy = strategy
        self.load_data(file_path)
        self.build_graph()

    def load_data(self, file_path: str) -> None:
        try:
            self.df = pd.read_csv(file_path, dtype=str)
            self.df.dropna(inplace=True)

            time_cols = ["departure_time", "arrival_time"]
            for col in time_cols:
                self.df[col] = self.df[col].apply(self.clean_time).astype(str)
                self.df[col] = pd.to_datetime(self.df[col], format="%H:%M:%S").dt.time

            # Override departure_time and arrival_time with the adjusted times
            for i, row in self.df.iterrows():
                departure_time = row["departure_time"]
                arrival_time = row["arrival_time"]

                # Use 1900-01-01 as the default date
                departure_datetime = datetime.combine(datetime(1900, 1, 1), departure_time)
                arrival_datetime = datetime.combine(datetime(1900, 1, 1), arrival_time)

                # If arrival_time >= departure_time, add +1 day to arrival_time
                if arrival_time >= departure_time:
                    departure_datetime += timedelta(days=1)

                arrival_datetime += timedelta(days=1)

                # Update the columns directly
                self.df.at[i, "departure_time"] = departure_datetime
                self.df.at[i, "arrival_time"] = arrival_datetime

            # Ensure that both 'departure_time' and 'arrival_time' are datetime objects
            self.df["departure_time"] = pd.to_datetime(self.df["departure_time"])
            self.df["arrival_time"] = pd.to_datetime(self.df["arrival_time"])

            # Calculate duration as the difference between the two datetime objects
            self.df["duration"] = (self.df["arrival_time"] - self.df["departure_time"]).dt.total_seconds()

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
        for _, row in self.df.iterrows():
            start, end = row["start_stop"], row["end_stop"]

            self.graph.add_edge(start, end)
            trips = self.graph[start][end].setdefault("trips", [])

            trip = {
                "departure_time": row["departure_time"],
                "arrival_time": row["arrival_time"],
                "duration": row["duration"],
                "line": row["line"],
            }

            insort(trips, trip, key=lambda t: t["arrival_time"])

    def apply_duration(self):
        return self.df.apply(
            lambda row: (
                datetime.combine(datetime.min, row["arrival_time"])
                - datetime.combine(datetime.min, row["departure_time"])
            ).seconds,
            axis=1,
        )

    def get_graph_stats(self) -> str:
        return (
            f"Graph Stats:\n" f" - Nodes: {self.graph.number_of_nodes()}\n" f" - Edges: {self.graph.number_of_edges()}"
        )

    def find_shortest_path(
        self, start: str, end: str, start_time_at_stop: datetime, heuristic=None
    ) -> tuple[float, list[str]]:
        cost, path = self.strategy.find_path(self.graph, start, end, start_time_at_stop, heuristic)
        return cost, path
