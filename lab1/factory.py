from heuristics import HaversineMaxTimeHeuristic, \
    ManhattanTransferHeuristic
from lab1.path_finding_strategies import (
    AStarOptimizedTimeStrategy,
    AStarOptimizedTransferStrategy,
    AStarTransfersStrategy,
    DijkstraTimeStrategy,
    DijkstraTransfersStrategy,
    PathfindingStrategy,
)
from lab1.path_finding_strategies.a_star_time import AStarTimeStrategy
from trip_selection_strategies import TimeBasedBestTripSelection, TransferBasedBestTripSelection


class StrategyFactory:
    @staticmethod
    def create_strategy(strategy_name: str) -> PathfindingStrategy:
        if strategy_name == "AStarTimeStrategy":
            return AStarTimeStrategy(TimeBasedBestTripSelection(), HaversineMaxTimeHeuristic())
        elif strategy_name == "AStarTransfersStrategy":
            return AStarTransfersStrategy(TransferBasedBestTripSelection(), ManhattanTransferHeuristic())
        elif strategy_name == "DijkstraTimeStrategy":
            return DijkstraTimeStrategy(TimeBasedBestTripSelection())
        elif strategy_name == "DijkstraTransfersStrategy":
            return DijkstraTransfersStrategy(TransferBasedBestTripSelection())
        elif strategy_name == "AStarOptimizedTimeStrategy":
            return AStarOptimizedTimeStrategy(TimeBasedBestTripSelection(), HaversineMaxTimeHeuristic())
        elif strategy_name == "AStarOptimizedTransferStrategy":
            return AStarOptimizedTransferStrategy(TransferBasedBestTripSelection(), ManhattanTransferHeuristic())
        else:
            raise ValueError(f"Unknown strategy: {strategy_name}")
