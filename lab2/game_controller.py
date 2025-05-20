import asyncio
import json

from clobber_game import ClobberGame
from message_types import MessageType


class GameController:
    def __init__(self, game: ClobberGame, display: bool = True):
        self.game = game
        self.display = display

    def play(self):
        while not self.game.is_game_over():
            if self.display:
                self.game.board.display()

            player = self.game.get_current_player()
            move = player.get_move(self.game)

            if move is None:
                print(f"Player {self.game.current_player} resigned!")
                break

            if not self.game.make_move(move):
                print("Invalid move! Try again.")

        if self.display:
            self.game.board.display()

        winner = self.game.get_winner()
        print(f"Game over. Winner: {winner}" if winner else "Game ended without winner.")
        node_count = self.game.players["B"].node_count
        print(f"Node processed: {node_count}")
        return winner


class WebSocketGameController:
    def __init__(self, game: ClobberGame, sockets: dict):
        self.game = game
        self.sockets = sockets  # {'B': ws1, 'W': ws2}

    async def play(self):
        while not self.game.is_game_over():
            player = self.game.get_current_player()

            if hasattr(player, "get_move") and asyncio.iscoroutinefunction(player.get_move):
                move = await player.get_move(self.game)
            else:
                move = player.get_move(self.game)

            if move is None:
                print(f"Player {self.game.current_player} resigned!")
                break

            if not self.game.make_move(move):
                print("Invalid move! Try again.")
                continue

            await self.broadcast(
                {
                    "type": MessageType.MOVE_MADE,
                    "move": move,
                    "board": self.game.board.board,
                    "current_player": self.game.current_player,
                    "game_over": self.game.is_game_over(),
                }
            )

        winner = self.game.get_winner()
        await self.broadcast({"type": MessageType.GAME_OVER, "winner": winner, "board": self.game.board.board})
        return winner

    async def broadcast(self, message: dict):
        for ws in self.sockets.values():
            if ws:
                try:
                    await ws.send(json.dumps(message))
                except Exception as e:
                    print(f"WebSocket error: {e}")
