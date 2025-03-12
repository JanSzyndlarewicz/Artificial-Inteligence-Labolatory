import logging
import time
from datetime import datetime, timedelta

from config import DATA_FILE_PATH
from graph_builder import TransitGraph
from factory import StrategyFactory

logger = logging.getLogger(__name__)


def main():
    start_time = time.time()

    # Collect user input
    # start_stop = input("Enter start stop: ")
    # end_stop = input("Enter end stop: ")
    # optimization_criteria = input("Enter optimization (t for time, p for transfers): ")
    # start_time_at_stop = input("Enter start time (HH:MM:SS): ")

    start_stop = "Wojnów"
    end_stop = "Rynek"
    optimization_criteria = "p"
    start_time_at_stop = "12:43:00"

    # # Choose the correct strategy based on user input
    # if optimization_criteria == "t":
    #     strategy = AStarTimeStrategy()
    #     heuristic_func = heuristic
    # elif optimization_criteria == "p":
    #     strategy = AStarTransfersStrategy()
    #     heuristic_func = heuristic
    # elif optimization_criteria == "dt":
    #     strategy = DijkstraTimeStrategy()
    #     heuristic_func = None
    # elif optimization_criteria == "dp":
    #     strategy = DijkstraTransfersStrategy()
    #     heuristic_func = None
    # else:
    #     raise ValueError("Invalid optimization criteria")
    #
    # # Create the TransitGraph object with the selected strategy
    # transit_graph = TransitGraph(file_path, strategy)
    #
    # # Find the shortest path
    # start_time_at_stop_dt = datetime.strptime(start_time_at_stop, "%H:%M:%S") + timedelta(days=1)
    # print(start_time_at_stop_dt)
    # cost, path = transit_graph.find_shortest_path(start_stop, end_stop, start_time_at_stop_dt, heuristic_func=heuristic_func)
    #
    # # Output the results
    # print(f"Cost: {cost}, Path: {path}")
    # print(f"Computation time: {time.time() - start_time:.6f} seconds")

    strategy_names = [
        "DijkstraTimeStrategy",
        "DijkstraTransfersStrategy",
        "AStarTimeStrategy",
        "AStarTransfersStrategy",
        "AStarOptimizedTimeStrategy",
        "AStarOptimizedTransferStrategy",
    ]

    strategies = [(name, StrategyFactory.create_strategy(name)) for name in strategy_names]

    start_time_at_stop_dt = datetime.strptime(start_time_at_stop, "%H:%M:%S") + timedelta(days=1)
    transit_graph = TransitGraph(DATA_FILE_PATH)
    for strategy_name, strategy in strategies:
        logger.info(f"Using strategy: {strategy_name}")
        cost, path = transit_graph.find_shortest_path(strategy, start_stop, end_stop, start_time_at_stop_dt)
        logger.info(f"Cost: {cost}, Path: {path}")
        logger.info(f"Computation timestamp: {time.time() - start_time:.6f} seconds")
        logger.info("------------------------")


if __name__ == "__main__":
    main()
