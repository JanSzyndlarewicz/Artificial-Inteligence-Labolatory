from heuristics import EuclideanDistanceHeuristic
from path_finder import AStarTimeStrategy, AStarTransfersStrategy, DijkstraTimeStrategy, DijkstraTransfersStrategy
from trip_selection_strategies import TimeBasedBestTripSelection, TransferBasedBestTripSelection


class StrategyFactory:
    @staticmethod
    def create_strategy(strategy_name: str):
        if strategy_name == "AStarTimeStrategy":
            return AStarTimeStrategy(TimeBasedBestTripSelection(), EuclideanDistanceHeuristic())
        elif strategy_name == "AStarTransfersStrategy":
            return AStarTransfersStrategy(TransferBasedBestTripSelection(), EuclideanDistanceHeuristic())
        elif strategy_name == "DijkstraTimeStrategy":
            return DijkstraTimeStrategy(TimeBasedBestTripSelection())
        elif strategy_name == "DijkstraTransfersStrategy":
            return DijkstraTransfersStrategy(TransferBasedBestTripSelection())
        else:
            raise ValueError(f"Unknown strategy: {strategy_name}")
