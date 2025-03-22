from heuristics import HaversineDistanceHeuristic
from path_finding_strategies import (
    AStarOptimizedTimeStrategy,
    AStarOptimizedTransferStrategy,
    AStarTransfersStrategy,
    DijkstraTimeStrategy,
    DijkstraTransfersStrategy,
    PathfindingStrategy,
)
from path_finding_strategies.a_star_time import AStarTimeStrategy
from trip_selection_strategies import TimeBasedBestTripSelection, TransferBasedBestTripSelection


class StrategyFactory:
    @staticmethod
    def create_strategy(strategy_name: str) -> PathfindingStrategy:
        if strategy_name == "AStarTimeStrategy":
            return AStarTimeStrategy(TimeBasedBestTripSelection(), HaversineDistanceHeuristic())
        elif strategy_name == "AStarTransfersStrategy":
            return AStarTransfersStrategy(TransferBasedBestTripSelection(), HaversineDistanceHeuristic())
        elif strategy_name == "DijkstraTimeStrategy":
            return DijkstraTimeStrategy(TimeBasedBestTripSelection())
        elif strategy_name == "DijkstraTransfersStrategy":
            return DijkstraTransfersStrategy(TransferBasedBestTripSelection())
        elif strategy_name == "AStarOptimizedTimeStrategy":
            return AStarOptimizedTimeStrategy(TimeBasedBestTripSelection(), HaversineDistanceHeuristic())
        elif strategy_name == "AStarOptimizedTransferStrategy":
            return AStarOptimizedTransferStrategy(TransferBasedBestTripSelection(), HaversineDistanceHeuristic())
        else:
            raise ValueError(f"Unknown strategy: {strategy_name}")
