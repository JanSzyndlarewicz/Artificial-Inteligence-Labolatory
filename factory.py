from heuristics import EuclideanDistanceHeuristic
from path_finding_strategies import PathfindingStrategy, AStarTransfersStrategy, DijkstraTimeStrategy, \
    DijkstraTransfersStrategy, AStarOptimizedTimeStrategy, AStarOptimizedTransferStrategy
from path_finding_strategies.a_star_time import AStarTimeStrategy
from trip_selection_strategies import TimeBasedBestTripSelection, TransferBasedBestTripSelection


class StrategyFactory:
    @staticmethod
    def create_strategy(strategy_name: str) -> PathfindingStrategy:
        if strategy_name == "AStarTimeStrategy":
            return AStarTimeStrategy(TimeBasedBestTripSelection(), EuclideanDistanceHeuristic())
        elif strategy_name == "AStarTransfersStrategy":
            return AStarTransfersStrategy(TransferBasedBestTripSelection(), EuclideanDistanceHeuristic())
        elif strategy_name == "DijkstraTimeStrategy":
            return DijkstraTimeStrategy(TimeBasedBestTripSelection())
        elif strategy_name == "DijkstraTransfersStrategy":
            return DijkstraTransfersStrategy(TransferBasedBestTripSelection())
        elif strategy_name == "AStarOptimizedTimeStrategy":
            return AStarOptimizedTimeStrategy(TimeBasedBestTripSelection(), EuclideanDistanceHeuristic())
        elif strategy_name == "AStarOptimizedTransferStrategy":
            return AStarOptimizedTransferStrategy(TransferBasedBestTripSelection(), EuclideanDistanceHeuristic())
        else:
            raise ValueError(f"Unknown strategy: {strategy_name}")
