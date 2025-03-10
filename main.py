import time
from datetime import datetime, timedelta


from data_import import TransitGraph
from path_finder import AStarTimeStrategy, AStarTransfersStrategy, DijkstraTimeStrategy, DijkstraTransfersStrategy, \
    heuristic


def main():
    file_path = "data/connection_graph.csv"
    start_time = time.time()

    # Collect user input
    # start_stop = input("Enter start stop: ")
    # end_stop = input("Enter end stop: ")
    # optimization_criteria = input("Enter optimization (t for time, p for transfers): ")
    # start_time_at_stop = input("Enter start time (HH:MM:SS): ")

    start_stop = "PL. GRUNWALDZKI"
    end_stop = "Rynek"
    optimization_criteria = "dt"
    start_time_at_stop = "12:43:00"

    # Choose the correct strategy based on user input
    if optimization_criteria == "t":
        strategy = AStarTimeStrategy()
        heuristic_func = heuristic
    elif optimization_criteria == "p":
        strategy = AStarTransfersStrategy()
        heuristic_func = heuristic
    elif optimization_criteria == "dt":
        strategy = DijkstraTimeStrategy()
        heuristic_func = None
    elif optimization_criteria == "dp":
        strategy = DijkstraTransfersStrategy()
        heuristic_func = None
    else:
        raise ValueError("Invalid optimization criteria")

    # Create the TransitGraph object with the selected strategy
    transit_graph = TransitGraph(file_path, strategy)

    # Find the shortest path
    start_time_at_stop_dt = datetime.strptime(start_time_at_stop, "%H:%M:%S") + timedelta(days=1)
    print(start_time_at_stop_dt)
    cost, path = transit_graph.find_shortest_path(start_stop, end_stop, start_time_at_stop_dt, heuristic_func=heuristic_func)

    # Output the results
    print(f"Cost: {cost}, Path: {path}")
    print(f"Computation time: {time.time() - start_time:.6f} seconds")


if __name__ == "__main__":
    main()
