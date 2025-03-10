from datetime import datetime
import networkx as nx
import pandas as pd
from path_finder import PathfindingStrategy


class TransitGraph:
    def __init__(self, file_path: str, strategy: PathfindingStrategy) -> None:
        self.df = None
        self.graph = nx.DiGraph()  # Using DiGraph instead of MultiDiGraph
        self.strategy = strategy
        self.load_data(file_path)
        self.build_graph()

    def load_data(self, file_path: str) -> None:
        try:
            self.df = pd.read_csv(file_path, dtype=str)
            self.df.dropna(inplace=True)

            time_cols = ['departure_time', 'arrival_time']
            for col in time_cols:
                self.df[col] = self.df[col].apply(self.clean_time).astype(str)
                self.df[col] = pd.to_datetime(self.df[col], format='%H:%M:%S').dt.time

            self.df['duration'] = self.apply_duration()

        except Exception as e:
            raise ValueError(f"Error loading data: {e}")

    @staticmethod
    def clean_time(time_str: str) -> str:
        try:
            hours, minutes, seconds = map(int, time_str.split(':'))
            return f'{hours % 24:02}:{minutes:02}:{seconds:02}'
        except ValueError:
            raise ValueError(f"Invalid time format: {time_str}")

    def build_graph(self) -> None:
        for _, row in self.df.iterrows():
            start, end = row['start_stop'], row['end_stop']

            self.graph.add_edge(start, end)
            self.graph[start][end].setdefault('trips', []).append({
                'departure_time': str(row['departure_time']),
                'arrival_time': str(row['arrival_time']),
                'duration': row['duration'],
                'line': row['line'],
            })

    def apply_duration(self):
        return self.df.apply(
            lambda row: (datetime.combine(datetime.min, row['arrival_time']) - datetime.combine(datetime.min, row['departure_time'])).seconds,
            axis=1
        )

    def get_graph_stats(self) -> str:
        return (f"Graph Stats:\n"
                f" - Nodes: {self.graph.number_of_nodes()}\n"
                f" - Edges: {self.graph.number_of_edges()}")

    def find_shortest_path(self, start, end, start_time_at_stop, heuristic=None):
        cost, path = self.strategy.find_path(self.graph, start, end, start_time_at_stop, heuristic)
        return cost, path
