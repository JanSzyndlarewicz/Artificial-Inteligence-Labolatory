from abc import abstractmethod, ABC
from datetime import datetime


class BestTripSelectionStrategy(ABC):
    @abstractmethod
    def get_best_trip(self, trips: list[dict], arrival_time: datetime, current_line: str | None) -> dict | None:
        pass


class TimeBasedBestTripSelection(BestTripSelectionStrategy):
    def get_best_trip(self, trips: list[dict], arrival_time: datetime, current_line: str | None) -> dict | None:
        best_trip = None
        for trip in trips:
            if trip["departure_time"] >= arrival_time:
                if (
                    best_trip is None
                    or trip["departure_time"] < best_trip["departure_time"]
                    or (trip["departure_time"] == best_trip["departure_time"] and trip["line"] == current_line)
                ):
                    best_trip = trip
        return best_trip


class TransferBasedBestTripSelection(BestTripSelectionStrategy):
    def get_best_trip(self, trips: list[dict], arrival_time: datetime, current_line: str | None) -> dict | None:
        best_trip = None
        for trip in trips:
            if trip["departure_time"] >= arrival_time:
                if current_line is None or trip["line"] == current_line:
                    if best_trip is None or trip["departure_time"] < best_trip["departure_time"]:
                        best_trip = trip
                elif best_trip is None or trip["departure_time"] < best_trip["departure_time"]:
                    best_trip = trip
        return best_trip