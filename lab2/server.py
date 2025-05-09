import asyncio
import json

from clobber_game import ClobberGame
from game_controller import WebSocketGameController
from heuristics import HeuristicType
from player_factory import PlayerFactory
from players import WebSocketHumanPlayer


class WebSocketGameServer:
    def __init__(self):
        self.games = {}        # game_id -> ClobberGame
        self.players = {}      # websocket -> (game_id, color)
        self.waiting_players = asyncio.Queue()

        self.initial_board = [
            ['B', 'W', 'B', 'W', 'B'],
            ['W', 'B', 'W', 'B', 'W'],
            ['B', 'W', 'B', 'W', 'B'],
            ['W', 'B', 'W', 'B', 'W'],
            ['B', 'W', 'B', 'W', 'B'],
            ['W', 'B', 'W', 'B', 'W'],
        ]

    async def handle_client(self, websocket):
        try:
            async for message in websocket:
                data = json.loads(message)
                msg_type = data.get("type")

                if msg_type == "register":
                    await self.handle_registration(websocket, data)
                elif msg_type == "make_move":
                    await self.handle_move(websocket, data)
                elif msg_type == "create_game":
                    await self.handle_create_game(websocket, data)
        except Exception as e:
            print(f"Error: {e}")
            await self.handle_disconnect(websocket)

    async def handle_registration(self, websocket, data):
        player_type = data.get("player_type", "ws_human")
        opponent_type = data.get("opponent_type", "ws_human")
        depth = data.get("depth", 3)
        heuristic = HeuristicType[data.get("heuristic_type", "MOBILITY")]

        if opponent_type == "human":
            try:
                opponent_ws = self.waiting_players.get_nowait()

                player_b = PlayerFactory.create('ws_human', color='B', websocket=websocket)
                player_w = PlayerFactory.create('ws_human', color='W', websocket=opponent_ws)

                game = ClobberGame(self.initial_board, player_b, player_w)
                self.games[game.game_id] = game
                self.players[websocket] = (game.game_id, 'B')
                self.players[opponent_ws] = (game.game_id, 'W')

                # Notify both players
                await self._send_game_started(websocket, game, 'B')
                await self._send_game_started(opponent_ws, game, 'W')

                controller = WebSocketGameController(game, {'B': websocket, 'W': opponent_ws})
                asyncio.create_task(controller.play())

            except asyncio.QueueEmpty:
                await self.waiting_players.put(websocket)
                await websocket.send(json.dumps({"type": "waiting_for_opponent"}))

        else:  # opponent_type == "ai"
            player_b = PlayerFactory.create('ws_human', color='B', websocket=websocket)
            player_w = PlayerFactory.create('ai', color='W', depth=depth, heuristic_type=heuristic)

            game = ClobberGame(self.initial_board, player_b, player_w)
            self.games[game.game_id] = game
            self.players[websocket] = (game.game_id, 'B')

            await self._send_game_started(websocket, game, 'B', ai=True)
            controller = WebSocketGameController(game, {'B': websocket, 'W': None})
            asyncio.create_task(controller.play())

    async def _send_game_started(self, websocket, game, color, ai=False):
        await websocket.send(json.dumps({
            "type": "game_started",
            "color": color,
            "board": game.board.board,
            "game_id": game.game_id,
            "opponent_type": "ai" if ai else "human"
        }))

    async def handle_move(self, websocket, data):
        if websocket not in self.players:
            return

        game_id, color = self.players[websocket]
        game = self.games.get(game_id)
        if not game or game.current_player != color:
            return

        try:
            move = tuple(tuple(m) for m in data["move"])
            player = game.players[color]
            if isinstance(player, WebSocketHumanPlayer):
                await player.move_queue.put(move)
        except Exception as e:
            print(f"Invalid move format: {e}")
            await websocket.close(code=1011, reason="Invalid move format")

    async def handle_create_game(self, websocket, data):
        board = data.get('board', self.initial_board)
        p1_type = data.get('player1_type', 'ws_human')
        p2_type = data.get('player2_type', 'ai')
        h1 = HeuristicType[data.get('heuristic1', 'MOBILITY')]
        h2 = HeuristicType[data.get('heuristic2', 'MOBILITY')]
        d1 = data.get('depth1', 3)
        d2 = data.get('depth2', 3)

        player1 = PlayerFactory.create(p1_type, color='B', depth=d1, heuristic_type=h1, websocket=websocket)
        player2 = PlayerFactory.create(p2_type, color='W', depth=d2, heuristic_type=h2, websocket=websocket)

        game = ClobberGame(board, player1, player2)
        self.games[game.game_id] = game

        self.players[websocket] = (game.game_id, 'B') if p1_type == 'ws_human' else (game.game_id, 'W')

        await websocket.send(json.dumps({
            "type": "game_created",
            "game_id": game.game_id,
            "board": game.board.board,
            "player_color": 'B' if p1_type == 'ws_human' else 'W'
        }))

        ws1 = websocket if p1_type == 'ws_human' else None
        ws2 = websocket if p2_type == 'ws_human' and p1_type != 'ws_human' else None
        controller = WebSocketGameController(game, {'B': ws1, 'W': ws2})
        asyncio.create_task(controller.play())

    async def handle_disconnect(self, websocket):
        if websocket not in self.players:
            return

        game_id, color = self.players.pop(websocket)
        game = self.games.get(game_id)
        if not game:
            return

        opponent_color = 'W' if color == 'B' else 'B'
        opponent = game.players.get(opponent_color)

        if isinstance(opponent, WebSocketHumanPlayer):
            try:
                await opponent.websocket.send(json.dumps({"type": "opponent_disconnected"}))
            except:
                pass

        self.games.pop(game_id, None)
