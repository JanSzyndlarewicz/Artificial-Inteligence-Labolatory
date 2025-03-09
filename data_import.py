import time
from datetime import datetime

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd


class TransitGraph:
    def __init__(self, file_path):
        self.df = pd.read_csv(file_path)

        self.df['departure_time'] = self.df['departure_time'].apply(self.clean_time)
        self.df['arrival_time'] = self.df['arrival_time'].apply(self.clean_time)

        self.df['departure_time'] = pd.to_datetime(self.df['departure_time'], format='%H:%M:%S').dt.time
        self.df['arrival_time'] = pd.to_datetime(self.df['arrival_time'], format='%H:%M:%S').dt.time

        self.df['duration'] = self.df.apply(
            lambda row: (datetime.combine(datetime.min, row['arrival_time']) - datetime.combine(datetime.min, row[
                'departure_time'])).seconds,
            axis=1
        )

        print(self.df)
        print(self.df.dtypes)

        self.graph = nx.DiGraph()
        self.build_graph()

        # Print the number of connections in the graph
        print(f"Number of connections in the graph: {self.graph.number_of_edges()}")

        # Print the number of nodes in the graph
        print(f"Number of nodes in the graph: {self.graph.number_of_nodes()}")

        self.visualize_graph()

    @staticmethod
    def clean_time(time_str):
        hours, minutes, seconds = map(int, time_str.split(':'))
        hours = hours % 24
        return f'{hours:02}:{minutes:02}:{seconds:02}'


    def build_graph(self):
        """Creates a directed graph from the transit data."""
        for _, row in self.df.iterrows():
            self.graph.add_edge(
                row['start_stop'],
                row['end_stop'],
                departure_time=str(row['departure_time']),
                arrival_time=str(row['arrival_time']),
                duration=row['duration']
            )

    def shortest_path(self, start, end, weight='duration'):
        """Finds the shortest path based on the given weight (default: travel duration)."""
        try:
            return nx.shortest_path(self.graph, source=start, target=end, weight=weight)
        except nx.NetworkXNoPath:
            return f"No path found from {start} to {end}"

    def visualize_graph(self):
        """Visualizes the graph using NetworkX and Matplotlib."""
        plt.figure(figsize=(10, 6))
        pos = nx.spring_layout(self.graph)
        nx.draw(self.graph, pos, with_labels=True, node_color='skyblue', edge_color='gray', node_size=2000, font_size=10)
        plt.show()


def main():
    file_path = 'data/connection_graph.csv'

    start_time = time.time()
    graph = TransitGraph(file_path)
    end_time = time.time()

    initialization_time = end_time - start_time
    print(f"Graph initialization time: {initialization_time:.6f} seconds")

    # start_stop = input("Podaj przystanek początkowy: ")
    # end_stop = input("Podaj przystanek końcowy: ")
    # criterion = input("Podaj kryterium optymalizacyjne (t - czas, p - przesiadki): ")
    # start_time = input("Podaj czas pojawienia się na przystanku początkowym (HH:MM:SS): ")
    # start_time = datetime.strptime(start_time, '%H:%M:%S').time()
    #
    # start_t = time.time()
    #
    # if criterion == 't':
    #     result = graph.dijkstra(start_stop, end_stop, start_time)
    # else:
    #     result = graph.dijkstra(start_stop, end_stop, start_time)  # Placeholder for A* implementation
    #
    # end_t = time.time()
    #
    # if result:
    #     print("Harmonogram przejazdu:")
    #     for stop, cost, arr_time in result:
    #         print(f"{stop} - koszt: {cost} sek, czas przyjazdu: {arr_time}")
    # else:
    #     print("Brak możliwego połączenia.")
    #
    # print(f"Czas obliczeń: {end_t - start_t:.6f} s")


if __name__ == "__main__":
    main()
