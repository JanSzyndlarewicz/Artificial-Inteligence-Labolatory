import logging
import time
from datetime import datetime, timedelta

from config import DATA_FILE_PATH
from factory import StrategyFactory
from graph_builder import TransitGraph
from transit_optimizer import TransitOptimizer

logger = logging.getLogger(__name__)


def main():
    start_time = time.time()

    # Collect user input
    # start_stop = input("Enter start stop: ")
    # end_stop = input("Enter end stop: ")
    # optimization_criteria = input("Enter optimization (t for time, p for transfers): ")
    # start_time_at_stop = input("Enter start time (HH:MM:SS): ")

    # start_stop = "Wojnów"
    # end_stop = "Rynek"
    # # start_time_at_stop = "12:43:00"
    # start_stop = "PL. GRUNWALDZKI"
    # end_stop = "EPI"
    start_stop = "PL. GRUNWALDZKI"
    end_stop = "Rynek"
    optimization_criteria = "p"
    start_time_at_stop = "16:30:00"

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

    transit_graph = TransitGraph(DATA_FILE_PATH)
    transit_optimizer = TransitOptimizer(StrategyFactory.create_strategy("AStarTimeStrategy"))

    start = "PL. GRUNWALDZKI"
    stops = ["Wojnów", "PL. GRUNWALDZKI", "EPI", "Rynek", "Kamieńskiego"]

    start_time_at_stop = "16:30:00"
    date_str = "02.01.1900"

    # Convert start_time_at_stop to a time object
    start_time_obj = datetime.strptime(start_time_at_stop, "%H:%M:%S").time()
    # Convert time to float representing seconds since midnight
    start_time_at_stop_float = timedelta(
        hours=start_time_obj.hour, minutes=start_time_obj.minute, seconds=start_time_obj.second
    ).total_seconds()

    start_time_at_stop = datetime(1900, 1, 2, 16, 30, 0)

    transit_graph.optimize_transit(transit_optimizer, start, stops, start_time_at_stop)

    exit(0)

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

    for strategy_name, strategy in strategies:
        logger.info(f"Using strategy: {strategy_name}")
        cost, path = transit_graph.find_shortest_path(strategy, start_stop, end_stop, start_time_at_stop_dt)
        logger.info(f"Cost: {cost}, Path: {path}")
        logger.info(f"Computation timestamp: {time.time() - start_time:.6f} seconds")
        logger.info("------------------------")


# def main():
#     start_time = time.time()
#
#     transit_graph = TransitGraph(DATA_FILE_PATH)
#
#     service_type = input("Choose service type (1: fastest route A->B, 2: fastest path through set of points): ")
#
#     if service_type == "1":
#         start_stop = input("Enter start stop: ")
#         end_stop = input("Enter end stop: ")
#         optimization_criteria = input("Enter optimization (t for time, p for transfers): ")
#         start_time_at_stop = input("Enter start time (HH:MM:SS): ")
#
#         strategy = StrategyFactory.create_strategy(
#             "AStarTimeStrategy" if optimization_criteria == "t" else "AStarTransfersStrategy")
#         start_time_at_stop_dt = datetime.strptime(start_time_at_stop, "%H:%M:%S") + timedelta(days=1)
#         cost, path = transit_graph.find_shortest_path(strategy, start_stop, end_stop, start_time_at_stop_dt)
#
#         print(f"Cost: {cost}, Path: {path}")
#
#     elif service_type == "2":
#         start_stop = input("Enter start stop: ")
#         stops = input("Enter stops to visit (comma-separated): ").split(",")
#         start_time_at_stop = input("Enter start time (HH:MM:SS): ")
#
#         transit_optimizer = TransitOptimizer(StrategyFactory.create_strategy("AStarTimeStrategy"))
#         start_time_at_stop_dt = datetime.strptime(start_time_at_stop, "%H:%M:%S")
#         transit_graph.optimize_transit(transit_optimizer, start_stop, stops, start_time_at_stop_dt)
#
#     else:
#         print("Invalid choice.")
#         return
#
#     print(f"Computation time: {time.time() - start_time:.6f} seconds")

if __name__ == "__main__":
    main()
