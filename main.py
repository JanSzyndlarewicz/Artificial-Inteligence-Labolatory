import logging
import sys
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

    start_stop = "Wojnów"
    end_stop = "Rynek"
    # # start_time_at_stop = "12:43:00"
    # start_stop = "PL. GRUNWALDZKI"
    # end_stop = "EPI"
    # start_stop = "LEŚNICA"
    # end_stop = "C.H. Aleja Bielany"
    optimization_criteria = "p"
    start_time_at_stop = "12:34:00"

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

    start = "Pl. Grunwaldzki"
    stops = ["Wojnów", "Epi", "Rynek", "Kamieńskiego"]

    start_time_at_stop = datetime(1900, 1, 2, 16, 30, 0)

    transit_graph.optimize_transit(transit_optimizer, start, stops, start_time_at_stop)

    #
    # strategy_names = [
    #     "DijkstraTimeStrategy",
    #     "DijkstraTransfersStrategy",
    #     "AStarTimeStrategy",
    #     "AStarTransfersStrategy",
    #     "AStarOptimizedTimeStrategy",
    #     "AStarOptimizedTransferStrategy",
    # ]
    #
    # strategies = [(name, StrategyFactory.create_strategy(name)) for name in strategy_names]
    #
    # start_time_at_stop_dt = datetime.strptime(start_time_at_stop, "%H:%M:%S") + timedelta(days=1)
    #
    # for strategy_name, strategy in strategies:
    #     logger.info(f"Using strategy: {strategy_name}")
    #     cost, path = transit_graph.find_shortest_path(strategy, start_stop, end_stop, start_time_at_stop_dt)
    #     logger.info(f"Cost: {cost}, Path: {path}")
    #     logger.info(f"Computation timestamp: {time.time() - start_time:.6f} seconds")
    #     logger.info("------------------------")


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

def parse_and_print_schedule(cost, path, execution_time, cost_type="t"):
    schedule = []
    current_line = None
    start_stop = None
    start_time_str = None

    for segment in path:
        departure_time = segment['departure_time'].strftime("%H:%M:%S")
        if segment['line'] != current_line:
            if current_line is not None:
                schedule.append(f"Linia {current_line}: {start_time_str} - {start_stop} -> {departure_time} - {segment['from']}")
            current_line = segment['line']
            start_stop = segment['from']
            start_time_str = departure_time

    last_segment = path[-1]
    arrival_time = last_segment['arrival_time'].strftime("%H:%M:%S")
    schedule.append(f"Linia {current_line}: {start_time_str} - {start_stop} -> {arrival_time} - {last_segment['to']}")

    for entry in schedule:
        print(entry)

    print(f"Cost type: {cost_type}", file=sys.stderr)
    if cost_type == "t":
        print(f"Koszt: {cost} sekund, czyli {cost/60} minut", file=sys.stderr)
    elif cost_type == "p":
        print(f"Koszt: {cost} środków transportu, czyli {cost-1} przesiadki", file=sys.stderr)

    print(f"Czas obliczeń: {execution_time:.2f} s", file=sys.stderr)


def main_menu():
    service_type = "1"
    #service_type = input("Choose service type (1: single trip, 2: transit optimizer): ")

    if service_type == "1":
        # start_stop = input("Enter start stop: ")
        # end_stop = input("Enter end stop: ")
        # optimization_criteria = input("Enter optimization (t for time, p for transfers): ")
        # start_time_at_stop = input("Enter start time (HH:MM:SS): ")
        start_stop = "Wojnów"
        end_stop = "Epi"
        optimization_criteria = "p"
        start_time_at_stop = "16:20:00"

        if optimization_criteria == "t":
            strategy_name = input("Choose algorithm (1: Dijkstra, 2: A*, 3: A* optimized): ")
            if strategy_name == "1":
                strategy_name = "DijkstraTimeStrategy"
            elif strategy_name == "2":
                strategy_name = "AStarTimeStrategy"
            elif strategy_name == "3":
                strategy_name = "AStarOptimizedTimeStrategy"
            else:
                raise ValueError("Invalid algorithm choice")
        elif optimization_criteria == "p":
            strategy_name = input("Choose algorithm (1: Dijkstra, 2: A*, 3: A* optimized): ")
            if strategy_name == "1":
                strategy_name = "DijkstraTransfersStrategy"
            elif strategy_name == "2":
                strategy_name = "AStarTransfersStrategy"
            elif strategy_name == "3":
                strategy_name = "AStarOptimizedTransferStrategy"
            else:
                raise ValueError("Invalid algorithm choice")
        else:
            raise ValueError("Invalid optimization criteria")

        strategy = StrategyFactory.create_strategy(strategy_name)
        transit_graph = TransitGraph(DATA_FILE_PATH)
        start_time_at_stop_dt = datetime.strptime(start_time_at_stop, "%H:%M:%S") + timedelta(days=1)
        start_time = time.time()
        path, cost = transit_graph.find_shortest_path(strategy, start_stop, end_stop, start_time_at_stop_dt)
        execution_time = time.time() - start_time

        parse_and_print_schedule(cost, path, execution_time, optimization_criteria)

    elif service_type == "2":
        start_stop = "Pl. Grunwaldzki"
        stops = ["Wojnów", "Epi", "Rynek", "Kamieńskiego"]
        start_time_at_stop = "16:30:00"
        optimization_criteria = "t"
        # start_stop = input("Enter start stop: ")
        # stops = input("Enter stops to visit (comma-separated): ").split(",")
        # start_time_at_stop = input("Enter start time (HH:MM:SS): ")

        transit_optimizer = TransitOptimizer(StrategyFactory.create_strategy("AStarTimeStrategy"))
        start_time_at_stop_dt = datetime.strptime(start_time_at_stop, "%H:%M:%S") + timedelta(days=1)
        transit_graph = TransitGraph(DATA_FILE_PATH)
        start_time = time.time()
        route, path, cost = transit_graph.optimize_transit(transit_optimizer, start_stop, stops, start_time_at_stop_dt)
        print(route)
        execution_time = time.time() - start_time

        parse_and_print_schedule(cost, path, execution_time, optimization_criteria)

    else:
        print("Invalid choice.")
        return

    print(f"Computation time: {execution_time:.6f} seconds")

if __name__ == "__main__":
    main_menu()
