from abc import ABC, abstractmethod
from typing import Optional, Tuple, List, Protocol
import asyncio
import json

from message_types import MessageType

Move = Tuple[Tuple[int, int], Tuple[int, int]]


# === Interfejsy ===



class Player(ABC):
    def __init__(self, color: str):
        self.color = color

    @abstractmethod
    def get_move(self, game) -> Optional[Move]:
        pass


class AsyncPlayer(Player):
    @abstractmethod
    async def get_move(self, game) -> Optional[Move]:
        pass


# === Implementacje graczy ===

class HumanPlayer(Player):
    def get_move(self, game) -> Optional[Move]:
        valid_moves = game.get_valid_moves()
        while True:
            try:
                user_input = input(f"Player {self.color} - enter move (row1 col1 row2 col2): ").strip()
                if user_input.lower() == "quit":
                    return None
                r1, c1, r2, c2 = map(int, user_input.split())
                move = ((r1, c1), (r2, c2))
                if move in valid_moves:
                    return move
                print("Invalid move, try again.")
            except ValueError:
                print("Invalid input. Please enter 4 numbers separated by spaces.")


class WebSocketHumanPlayer(AsyncPlayer):
    def __init__(self, color: str, websocket):
        super().__init__(color)
        self.websocket = websocket
        self.move_queue = asyncio.Queue()

    async def get_move(self, game) -> Optional[Move]:
        valid_moves = game.get_valid_moves()
        await self.websocket.send(json.dumps({
            "type": MessageType.REQUEST_MOVE,
            "valid_moves": valid_moves,
            "board": game.board.board,
            "current_player": self.color
        }))
        move = await self.move_queue.get()
        return move


class AIPlayer(Player):
    def __init__(self, color: str, depth: int = 3, heuristic=None):
        super().__init__(color)
        self.depth = depth
        self.heuristic = heuristic
        self.history = []
        self.learned_weights = {
            'center': 1.0,
            'mobility': 1.0,
            'aggression': 1.0,
            'defense': 1.0
        }

    def get_move(self, game) -> Optional[Move]:
        print(f"AI {self.color} is thinking...")
        best_move = None
        best_value = float('-inf')
        alpha, beta = float('-inf'), float('inf')

        for move in game.get_valid_moves():
            new_game = game.copy()
            new_game.make_move(move)
            move_value = self.minimax(new_game, self.depth - 1, alpha, beta, False)

            if move_value > best_value or (move_value == best_value and best_move is None):
                best_value = move_value
                best_move = move

            alpha = max(alpha, best_value)

        return best_move

    def minimax(self, game, depth: int, alpha: float, beta: float, maximizing: bool) -> float:
        if depth == 0 or game.is_game_over():
            return self.heuristic(self, game)

        best = float('-inf') if maximizing else float('inf')
        compare = max if maximizing else min

        for move in game.get_valid_moves():
            new_game = game.copy()
            new_game.make_move(move)
            eval_ = self.minimax(new_game, depth - 1, alpha, beta, not maximizing)

            best = compare(best, eval_)
            if maximizing:
                alpha = max(alpha, eval_)
            else:
                beta = min(beta, eval_)
            if beta <= alpha:
                break

        return best
