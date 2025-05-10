import asyncio
import json

from clobber_game import ClobberGame
from game_controller import WebSocketGameController
from message_types import MessageType
from player_factory import PlayerFactory
from players import WebSocketPlayer


class WebSocketGameServer:
    def __init__(self):
        self.games = {}
        self.players = {}
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

                if msg_type == MessageType.REGISTER:
                    await self.handle_registration(websocket)
                elif msg_type == MessageType.MAKE_MOVE:
                    await self.handle_move(websocket, data)
                elif msg_type == MessageType.CREATE_GAME:
                    await self.handle_create_game(websocket, data)
        except Exception as e:
            print(f"Error: {e}")
            await self.handle_disconnect(websocket)

    async def handle_registration(self, websocket):
        try:
            opponent_ws = self.waiting_players.get_nowait()

            player_b = self._create_player('websocket', 'B', websocket=websocket)
            player_w = self._create_player('websocket', 'W', websocket=opponent_ws)
            game = self.create_game(self.initial_board, player_b, player_w)

            self.players[websocket] = (game.game_id, 'B')
            self.players[opponent_ws] = (game.game_id, 'W')

            await self._send_game_started(websocket, game, 'B')
            await self._send_game_started(opponent_ws, game, 'W')
            self._run_game_controller(game, {'B': websocket, 'W': opponent_ws})

        except asyncio.QueueEmpty:
            await self.waiting_players.put(websocket)
            await websocket.send(json.dumps({"type": MessageType.WAITING}))


    def _create_player(self, player_type, color, websocket=None, depth=3, heuristic=None):
        return PlayerFactory.create(player_type, color=color, websocket=websocket, depth=depth,
                                    heuristic_type=heuristic)

    def create_game(self, board, player_b, player_w):
        game = ClobberGame(board, player_b, player_w)
        self.games[game.game_id] = game
        return game

    def _run_game_controller(self, game, sockets):
        controller = WebSocketGameController(game, sockets)
        asyncio.create_task(controller.play())

    async def _send_game_started(self, websocket, game, color, ai=False):
        await websocket.send(json.dumps({
            "type": MessageType.GAME_STARTED,
            "color": color,
            "board": game.board.board,
            "game_id": game.game_id,
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
            if isinstance(player, WebSocketPlayer):
                await player.move_queue.put(move)
        except Exception as e:
            print(f"Invalid move format: {e}")
            await websocket.close(code=1011, reason="Invalid move format")

    async def handle_create_game(self, websocket, data):
        board = data.get('board', self.initial_board)
        p1_type = data.get('player1_type', 'websocket')
        p2_type = data.get('player2_type', 'ai')

        game = ClobberGame(board)
        self.games[game.game_id] = game
        self.players[websocket] = (game.game_id, 'B') if p1_type == 'websocket' else (game.game_id, 'W')

        await websocket.send(json.dumps({
            "type": MessageType.GAME_CREATED,
            "game_id": game.game_id,
            "board": game.board.board,
            "player_color": 'B' if p1_type == 'websocket' else 'W'
        }))

        ws1 = websocket if p1_type == 'websocket' else None
        ws2 = websocket if p2_type == 'websocket' and p1_type != 'websocket' else None
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

        if isinstance(opponent, WebSocketPlayer):
            try:
                await opponent.websocket.send(json.dumps({"type": MessageType.OPPONENT_DISCONNECTED}))
            except:
                pass

        self.games.pop(game_id, None)
