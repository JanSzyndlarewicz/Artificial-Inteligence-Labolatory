import time
from datetime import datetime
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd


class TransitGraph:
    def __init__(self, file_path: str) -> None:
        self.df = None
        self.graph = nx.MultiDiGraph()
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

            self.df['duration'] = self.df.apply(
                lambda row: (datetime.combine(datetime.min, row['arrival_time']) - datetime.combine(datetime.min, row[
                    'departure_time'])).seconds,
                axis=1
            )
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
            self.graph.add_edge(
                row['start_stop'],
                row['end_stop'],
                key=(row['departure_time'], row['arrival_time']),
                departure_time=str(row['departure_time']),
                arrival_time=str(row['arrival_time']),
                duration=row['duration']
            )

    def get_graph_stats(self) -> str:
        return (f"Graph Stats:\n"
                f" - Nodes: {self.graph.number_of_nodes()}\n"
                f" - Edges: {self.graph.number_of_edges()}")


class GraphVisualizer:
    @staticmethod
    def visualize(graph: nx.MultiDiGraph) -> None:
        plt.figure(figsize=(10, 6))
        pos = nx.spring_layout(graph)
        nx.draw(graph, pos, with_labels=True, node_color='skyblue', edge_color='gray', node_size=2000, font_size=10)
        plt.show()


def main():
    file_path = 'data/connection_graph.csv'
    start_time = time.time()

    transit_graph = TransitGraph(file_path)
    print(transit_graph.get_graph_stats())

    GraphVisualizer.visualize(transit_graph.graph)
    print(f"Graph initialization time: {time.time() - start_time:.6f} seconds")



if __name__ == "__main__":
    main()
