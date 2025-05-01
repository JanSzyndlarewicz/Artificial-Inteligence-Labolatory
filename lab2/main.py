import asyncio
import concurrent.futures
import concurrent.futures
import json
import math
from abc import ABC, abstractmethod
from copy import deepcopy
from enum import auto, Enum
from typing import Optional, Tuple, List, Callable, Dict
from uuid import uuid4

import websockets

Move = Tuple[Tuple[int, int], Tuple[int, int]]
Board = List[List[str]]


class HeuristicType(Enum):
    MOBILITY = auto()
    PIECE_COUNT = auto()
    CENTER_CONTROL = auto()
    AGGRESSIVE = auto()
    DEFENSIVE = auto()
    GOLDEN_SPOTS = auto()
    GROUP_MOBILITY = auto()
    ADAPTIVE = auto()
    OPPONENT_AWARE = auto()
    LEARNING = auto()
    HYBRID = auto()


class HeuristicFactory:
    _heuristics: Dict[HeuristicType, Callable] = {}

    @classmethod
    def register(cls, heuristic_type: HeuristicType):
        def decorator(func: Callable):
            cls._heuristics[heuristic_type] = func
            return func

        return decorator

    @classmethod
    def create(cls, heuristic_type: HeuristicType) -> Callable:
        return cls._heuristics.get(heuristic_type, cls._heuristics[HeuristicType.MOBILITY])


class Player(ABC):
    def __init__(self, color):
        self.color = color

    @abstractmethod
    def get_move(self, game):
        pass


class HumanPlayer(Player):
    def get_move(self, game):
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


class WebSocketHumanPlayer(Player):
    def __init__(self, color, websocket):
        super().__init__(color)
        self.websocket = websocket
        self.move_queue = asyncio.Queue()

    async def get_move(self, game):
        valid_moves = game.get_valid_moves()
        await self.websocket.send(json.dumps({
            "type": "request_move",
            "valid_moves": valid_moves,
            "board": game.board.board,
            "current_player": self.color
        }))

        # Wait for move from client
        move = await self.move_queue.get()
        return move


class AIPlayer(Player):
    def __init__(self, color: str, depth: int = 3, heuristic: Callable = None):
        super().__init__(color)
        self.depth = depth
        self.heuristic = heuristic or HeuristicFactory.create(HeuristicType.MOBILITY)
        self.opponent_style = None  # 'aggressive'/'defensive'/'balanced'
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

        if maximizing:
            max_eval = float('-inf')
            for move in game.get_valid_moves():
                new_game = game.copy()
                new_game.make_move(move)
                eval_ = self.minimax(new_game, depth - 1, alpha, beta, False)
                max_eval = max(max_eval, eval_)
                alpha = max(alpha, eval_)
                if beta <= alpha:
                    break
            return max_eval
        else:
            min_eval = float('inf')
            for move in game.get_valid_moves():
                new_game = game.copy()
                new_game.make_move(move)
                eval_ = self.minimax(new_game, depth - 1, alpha, beta, True)
                min_eval = min(min_eval, eval_)
                beta = min(beta, eval_)
                if beta <= alpha:
                    break
            return min_eval


class ClobberBoard:
    def __init__(self, board):
        self.board = board
        self.rows = len(board)
        self.cols = len(board[0])

    def copy(self):
        return ClobberBoard(deepcopy(self.board))

    def get_piece(self, row, col):
        return self.board[row][col]

    def set_piece(self, row, col, piece):
        self.board[row][col] = piece

    def display(self):
        print("  " + " ".join(str(i) for i in range(self.cols)))
        for i, row in enumerate(self.board):
            print(f"{i} " + " ".join(row))
        print()


class ClobberGame:
    def __init__(self, board, player1, player2):
        self.board = ClobberBoard(board)
        self.players = {'B': player1, 'W': player2}
        self.current_player = 'B'
        self.opponent = {'B': 'W', 'W': 'B'}
        self.game_id = str(uuid4())

    def copy(self):
        new_game = ClobberGame(deepcopy(self.board.board),
                               self.players['B'],
                               self.players['W'])
        new_game.current_player = self.current_player
        new_game.game_id = self.game_id
        return new_game

    def get_valid_moves(self):
        moves = []
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]

        for r in range(self.board.rows):
            for c in range(self.board.cols):
                if self.board.get_piece(r, c) == self.current_player:
                    for dr, dc in directions:
                        nr, nc = r + dr, c + dc
                        if (0 <= nr < self.board.rows and
                                0 <= nc < self.board.cols and
                                self.board.get_piece(nr, nc) == self.opponent[self.current_player]):
                            moves.append(((r, c), (nr, nc)))
        return moves

    def make_move(self, move):
        if move not in self.get_valid_moves():
            return False

        (r1, c1), (r2, c2) = move
        self.board.set_piece(r2, c2, self.current_player)
        self.board.set_piece(r1, c1, '_')
        self.switch_player()
        return True

    def switch_player(self):
        self.current_player = self.opponent[self.current_player]

    def is_game_over(self):
        return len(self.get_valid_moves()) == 0

    def play(self) -> str:
        while not self.is_game_over():
            self.board.display()
            current_player_obj = self.players[self.current_player]
            move = current_player_obj.get_move(self)

            if move is None:  # Player quit
                print(f"Player {self.current_player} resigned!")
                break

            if not self.make_move(move):
                print("Invalid move! Try again.")
                continue

        self.board.display()
        winner = self.opponent[self.current_player] if self.is_game_over() else None
        print(f"Game over. Winner: {winner}" if winner else "Game ended without winner.")
        return winner

    async def play_websocket(self, websocket1=None, websocket2=None) -> str:
        websockets = {'B': websocket1, 'W': websocket2}

        while not self.is_game_over():
            current_player_obj = self.players[self.current_player]

            if isinstance(current_player_obj, WebSocketHumanPlayer):
                move = await current_player_obj.get_move(self)
            else:
                move = current_player_obj.get_move(self)

            if move is None:  # Player quit
                print(f"Player {self.current_player} resigned!")
                break

            if not self.make_move(move):
                print("Invalid move! Try again.")
                continue

            # Notify both players about the move
            for color, ws in websockets.items():
                if ws:
                    try:
                        await ws.send(json.dumps({
                            "type": "move_made",
                            "move": move,
                            "board": self.board.board,
                            "current_player": self.current_player,
                            "game_over": self.is_game_over()
                        }))
                    except Exception as e:
                        print(f"Error sending to {color}: {e}")

        winner = self.opponent[self.current_player] if self.is_game_over() else None

        for color, ws in websockets.items():
            if ws:
                try:
                    await ws.send(json.dumps({
                        "type": "game_over",
                        "winner": winner,
                        "board": self.board.board
                    }))
                except Exception as e:
                    print(f"Error sending game over to {color}: {e}")

        return winner


def create_player(player_type: str, color: str, depth: int = 3,
                  heuristic_type: HeuristicType = HeuristicType.MOBILITY, websocket=None) -> Player:
    if player_type.lower() == 'human':
        return HumanPlayer(color)
    elif player_type.lower() == 'ws_human' and websocket:
        return WebSocketHumanPlayer(color, websocket)
    elif player_type.lower() == 'ai':
        heuristic_func = HeuristicFactory.create(heuristic_type)
        return AIPlayer(color, depth, heuristic_func)
    else:
        raise ValueError("Invalid player type. Choose 'human', 'ws_human' or 'ai'.")


def simulate_game(h1, h2, initial_board):
    player1 = create_player('ai', 'B', 5, h1)
    player2 = create_player('ai', 'W', 5, h2)
    game = ClobberGame(initial_board, player1, player2)
    winner = game.play()
    return h1, h2, winner


# WebSocket server implementation
class WebSocketGameServer:
    def __init__(self):
        self.games = {}  # game_id -> ClobberGame
        self.players = {}  # websocket -> (game_id, color)
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
        """Handle WebSocket client connections"""
        try:
            async for message in websocket:
                data = json.loads(message)

                if data['type'] == 'register':
                    await self.handle_registration(websocket, data)
                elif data['type'] == 'make_move':
                    await self.handle_move(websocket, data)
                elif data['type'] == 'create_game':
                    await self.handle_create_game(websocket, data)
        except websockets.exceptions.ConnectionClosed:
            await self.handle_disconnect(websocket)
        except Exception as e:
            print(f"Error handling client: {e}")
            try:
                await websocket.close(code=1011, reason=str(e))
            except:
                pass

    async def handle_registration(self, websocket, data):
        """Handle player registration and matchmaking"""
        player_type = data.get('player_type', 'human')
        heuristic_type = data.get('heuristic_type')
        depth = data.get('depth', 3)

        if player_type == 'ws_human':
            # Human player wants to play against another human or AI
            opponent_type = data.get('opponent_type', 'human')

            if opponent_type == 'human':
                # Try to pair with another human player
                try:
                    opponent_ws = self.waiting_players.get_nowait()
                    game = ClobberGame(self.initial_board,
                                       create_player('ws_human', 'B', websocket=websocket),
                                       create_player('ws_human', 'W', websocket=opponent_ws))
                    self.games[game.game_id] = game
                    self.players[websocket] = (game.game_id, 'B')
                    self.players[opponent_ws] = (game.game_id, 'W')

                    # Notify both players
                    await websocket.send(json.dumps({
                        "type": "game_started",
                        "color": "B",
                        "board": game.board.board,
                        "game_id": game.game_id
                    }))

                    await opponent_ws.send(json.dumps({
                        "type": "game_started",
                        "color": "W",
                        "board": game.board.board,
                        "game_id": game.game_id
                    }))

                    # Start the game
                    asyncio.create_task(game.play_websocket(websocket, opponent_ws))

                except asyncio.QueueEmpty:
                    # No opponent available, add to waiting list
                    await self.waiting_players.put(websocket)
                    await websocket.send(json.dumps({
                        "type": "waiting_for_opponent"
                    }))
            else:
                # Play against AI
                heuristic = HeuristicType[heuristic_type] if heuristic_type else HeuristicType.MOBILITY
                game = ClobberGame(self.initial_board,
                                   create_player('ws_human', 'B', websocket=websocket),
                                   create_player('ai', 'W', depth, heuristic))
                self.games[game.game_id] = game
                self.players[websocket] = (game.game_id, 'B')

                await websocket.send(json.dumps({
                    "type": "game_started",
                    "color": "B",
                    "board": game.board.board,
                    "game_id": game.game_id,
                    "opponent_type": "ai"
                }))

                # Start the game
                asyncio.create_task(game.play_websocket(websocket, None))

        elif player_type == 'ai':
            # AI vs AI game
            heuristic1 = HeuristicType[data.get('heuristic1', 'MOBILITY')]
            heuristic2 = HeuristicType[data.get('heuristic2', 'MOBILITY')]

            game = ClobberGame(self.initial_board,
                               create_player('ai', 'B', depth, heuristic1),
                               create_player('ai', 'W', depth, heuristic2))
            self.games[game.game_id] = game

            await websocket.send(json.dumps({
                "type": "ai_game_started",
                "game_id": game.game_id
            }))

            # Start the game and send updates to the observer
            await asyncio.create_task(self.run_ai_game(game, websocket))

    async def handle_move(self, websocket, data):
        """Handle player move"""
        if websocket not in self.players:
            return

        game_id, color = self.players[websocket]
        game = self.games.get(game_id)

        if not game or game.current_player != color:
            return

        try:
            # Convert the move into the expected format
            print(f"Received move from {data}")
            print(f"Received move: {data['move']}")
            move = tuple(tuple(m) for m in data['move'])  # Convert list of lists to tuple of tuples
            player = game.players[color]

            if isinstance(player, WebSocketHumanPlayer):
                await player.move_queue.put(move)
        except Exception as e:
            print(f"Error processing move: {e}")
            await websocket.close(code=1011, reason="Invalid move format")

    async def handle_create_game(self, websocket, data):
        """Handle custom game creation"""
        board = data.get('board', self.initial_board)
        player1_type = data.get('player1_type', 'human')
        player2_type = data.get('player2_type', 'human')
        heuristic1 = data.get('heuristic1')
        heuristic2 = data.get('heuristic2')
        depth1 = data.get('depth1', 3)
        depth2 = data.get('depth2', 3)

        player1 = create_player(
            player1_type, 'B', depth1,
            HeuristicType[heuristic1] if heuristic1 else None,
            websocket if player1_type == 'ws_human' else None
        )

        player2 = create_player(
            player2_type, 'W', depth2,
            HeuristicType[heuristic2] if heuristic2 else None,
            websocket if player2_type == 'ws_human' and player1_type != 'ws_human' else None
        )

        game = ClobberGame(board, player1, player2)
        self.games[game.game_id] = game

        if player1_type == 'ws_human':
            self.players[websocket] = (game.game_id, 'B')
        elif player2_type == 'ws_human':
            self.players[websocket] = (game.game_id, 'W')

        await websocket.send(json.dumps({
            "type": "game_created",
            "game_id": game.game_id,
            "board": game.board.board,
            "player_color": 'B' if player1_type == 'ws_human' else 'W' if player2_type == 'ws_human' else None
        }))

        # Start the game
        ws1 = websocket if player1_type == 'ws_human' else None
        ws2 = websocket if player2_type == 'ws_human' and player1_type != 'ws_human' else None
        asyncio.create_task(game.play_websocket(ws1, ws2))

    async def handle_disconnect(self, websocket):
        """Handle client disconnection"""
        if websocket in self.players:
            game_id, color = self.players.pop(websocket)
            game = self.games.get(game_id)

            if game:
                # Notify opponent if it's a human player
                opponent_color = 'W' if color == 'B' else 'B'
                opponent = game.players[opponent_color]

                if isinstance(opponent, WebSocketHumanPlayer) and opponent.websocket in self.players:
                    try:
                        await opponent.websocket.send(json.dumps({
                            "type": "opponent_disconnected"
                        }))
                    except:
                        pass

                # Remove the game
                self.games.pop(game_id, None)

    async def run_ai_game(self, game, websocket):
        """Run AI vs AI game and send updates to observer"""
        while not game.is_game_over():
            current_player_obj = game.players[game.current_player]
            move = current_player_obj.get_move(game)

            if not move:
                break

            game.make_move(move)

            try:
                await websocket.send(json.dumps({
                    "type": "ai_move",
                    "move": move,
                    "board": game.board.board,
                    "current_player": game.current_player
                }))
                await asyncio.sleep(1)  # Pause for visualization
            except:
                break

        winner = game.opponent[game.current_player] if game.is_game_over() else None

        try:
            await websocket.send(json.dumps({
                "type": "ai_game_over",
                "winner": winner,
                "board": game.board.board
            }))
        except:
            pass

async def connect_to_server():
    uri = input("Enter server URI (e.g., ws://localhost:8765): ")
    async with websockets.connect(uri) as websocket:
        print("Connected to server")

        # Example: Send a message to the server
        await websocket.send(json.dumps({"type": "join"}))
        print("Joined the game. Waiting for updates...")

        # Listen for messages from the server
        while True:
            message = await websocket.recv()
            print("Received:", message)

def main():
    initial_board = [
        ['B', 'W', 'B', 'W', 'B'],
        ['W', 'B', 'W', 'B', 'W'],
        ['B', 'W', 'B', 'W', 'B'],
        ['W', 'B', 'W', 'B', 'W'],
        ['B', 'W', 'B', 'W', 'B'],
        ['W', 'B', 'W', 'B', 'W'],
    ]

    print("Choose mode:")
    print("1. Local game")
    print("2. WebSocket server")
    print("3. Connect to WebSocket server")

    choice = input("Enter your choice (1-3): ")

    if choice == '1':
        # Local game
        print("Choose player types:")
        print("1. Human vs Human")
        print("2. Human vs AI")
        print("3. AI vs AI")
        print("4. AI vs AI (different heuristics)")
        print("5. AI Tournament (each heuristic vs each)")

        choice = input("Enter your choice (1-5): ")

        heuristic_types = list(HeuristicType)
        heuristic_descriptions = {
            HeuristicType.MOBILITY: "Mobility",
            HeuristicType.PIECE_COUNT: "Piece Count",
            HeuristicType.CENTER_CONTROL: "Center Control",
            HeuristicType.AGGRESSIVE: "Aggressiveness",
            HeuristicType.DEFENSIVE: "Defensive",
            HeuristicType.GOLDEN_SPOTS: "Golden Spots",
            HeuristicType.GROUP_MOBILITY: "Group Mobility",
            HeuristicType.ADAPTIVE: "Adaptive Hybrid",
            HeuristicType.OPPONENT_AWARE: "Style Aware",
            HeuristicType.LEARNING: "Learning",
            HeuristicType.HYBRID: "Hybrid"
        }

        if choice == '1':
            player1 = create_player('human', 'B')
            player2 = create_player('human', 'W')
            game = ClobberGame(initial_board, player1, player2)
            game.play()

        elif choice == '2':
            print("Choose AI heuristic:")
            for i, heuristic in enumerate(heuristic_types, 1):
                print(f"{i}. {heuristic_descriptions[heuristic]}")

            heuristic_choice = int(input("Enter heuristic choice: ")) - 1
            selected_heuristic = heuristic_types[heuristic_choice]

            player1 = create_player('human', 'B')
            player2 = create_player('ai', 'W', 3, selected_heuristic)
            game = ClobberGame(initial_board, player1, player2)
            game.play()

        elif choice == '3':
            player1 = create_player('ai', 'B')
            player2 = create_player('ai', 'W')
            game = ClobberGame(initial_board, player1, player2)
            game.play()

        elif choice == '4':
            print("Choose AI 1 heuristic:")
            for i, heuristic in enumerate(heuristic_types, 1):
                print(f"{i}. {heuristic_descriptions[heuristic]}")
            heuristic1 = heuristic_types[int(input("AI 1 choice: ")) - 1]

            print("Choose AI 2 heuristic:")
            for i, heuristic in enumerate(heuristic_types, 1):
                print(f"{i}. {heuristic_descriptions[heuristic]}")
            heuristic2 = heuristic_types[int(input("AI 2 choice: ")) - 1]

            player1 = create_player('ai', 'B', 3, heuristic1)
            player2 = create_player('ai', 'W', 3, heuristic2)
            game = ClobberGame(initial_board, player1, player2)
            game.play()

        elif choice == '5':
            results = {heuristic: {'wins': 0, 'losses': 0} for heuristic in heuristic_types}
            matches = [(h1, h2) for h1 in heuristic_types for h2 in heuristic_types if h1 != h2]

            print(f"\nRunning tournament with {len(matches)} matches...")
            MAX_WORKERS = 16

            with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
                futures = [executor.submit(simulate_game, h1, h2, initial_board)
                           for h1, h2 in matches]

                for i, future in enumerate(concurrent.futures.as_completed(futures), 1):
                    h1, h2, winner = future.result()
                    if winner == 'B':
                        results[h1]['wins'] += 1
                        results[h2]['losses'] += 1
                    elif winner == 'W':
                        results[h2]['wins'] += 1
                        results[h1]['losses'] += 1
                    print(f"Completed {i}/{len(matches)} matches")

            print("\n=== TOURNAMENT RESULTS ===")
            for heuristic, stats in results.items():
                print(f"{heuristic_descriptions[heuristic]}: {stats['wins']} wins, {stats['losses']} losses")

            best = max(results.items(), key=lambda x: x[1]['wins'])[0]
            print(f"\nBest heuristic: {heuristic_descriptions[best]} ({results[best]['wins']} wins)")

        else:
            print("Invalid choice. Defaulting to Human vs AI.")
            player1 = create_player('human', 'B')
            player2 = create_player('ai', 'W')
            game = ClobberGame(initial_board, player1, player2)
            game.play()

    elif choice == '2':
        # Start WebSocket server
        async def run_server():
            server = WebSocketGameServer()
            start_server = await websockets.serve(
                server.handle_client,
                "localhost",
                8765
            )
            print("WebSocket server started on ws://localhost:8765")
            await start_server.wait_closed()

        asyncio.run(run_server())

    elif choice == '3':
        # Connect to WebSocket server
        async def connect_to_server():
            uri = "ws://localhost:8765"  # Hardcoded URI
            async with websockets.connect(uri) as websocket:
                print("Connected to server")

                # Register as a human player
                await websocket.send(json.dumps({
                    "type": "register",
                    "player_type": "ws_human",
                    "opponent_type": "human"  # or "ai" to play against computer
                }))

                print("Waiting for game to start...")

                while True:
                    message = await websocket.recv()
                    data = json.loads(message)
                    print("Received message:", data)

                    if data['type'] == 'request_move':
                        # Display board and ask for move
                        print("Current board:")
                        for row in data['board']:
                            print(" ".join(row))
                        print(f"Valid moves: {data['valid_moves']}")
                        move = input("Enter your move (r1 c1 r2 c2): ")
                        print(f"Move entered: {move}")
                        r1, c1, r2, c2 = map(int, move.split())
                        await websocket.send(json.dumps({
                            "type": "make_move",
                            "move": [(r1, c1), (r2, c2)]
                        }))
                    elif data['type'] == 'move_made':
                        print("Move made:")
                        print(f"From: {data['move'][0]}, To: {data['move'][1]}")
                        print("Current board:")
                        for row in data['board']:
                            print(" ".join(row))
                    elif data['type'] == 'game_over':
                        print(f"Game over! Winner: {data['winner']}")
                        break
                    elif data['type'] == 'waiting_for_opponent':
                        print("Waiting for another player to join...")
                    elif data['type'] == 'game_started':
                        print(f"Game started! You are playing as {data['color']}")
                        print("Initial board:")
                        for row in data['board']:
                            print(" ".join(row))

        asyncio.get_event_loop().run_until_complete(connect_to_server())


# heuristucs

def default_heuristic(ai_player: AIPlayer, game) -> float:
    """Domyślna heurystyka oparta na mobilności"""
    return mobility_heuristic(ai_player, game)


@HeuristicFactory.register(HeuristicType.MOBILITY)
def mobility_heuristic(ai_player: AIPlayer, game) -> float:
    """Różnica w liczbie dostępnych ruchów między graczami"""
    if game.is_game_over():
        return math.inf if game.current_player != ai_player.color else -math.inf

    current_moves = len(game.get_valid_moves())
    game.switch_player()
    opponent_moves = len(game.get_valid_moves())
    game.switch_player()
    return current_moves - opponent_moves


@HeuristicFactory.register(HeuristicType.PIECE_COUNT)
def piece_count_heuristic(ai_player: AIPlayer, game) -> float:
    """Różnica w liczbie pozostałych pionków"""
    if game.is_game_over():
        return math.inf if game.current_player != ai_player.color else -math.inf

    current = sum(1 for row in game.board.board for cell in row if cell == ai_player.color)
    opponent = sum(1 for row in game.board.board for cell in row if cell == game.opponent[ai_player.color])
    return current - opponent


@HeuristicFactory.register(HeuristicType.CENTER_CONTROL)
def center_control_heuristic(ai_player: AIPlayer, game) -> float:
    """Wartość za kontrolę środkowych pól planszy"""
    center_r, center_c = game.board.rows / 2, game.board.cols / 2
    score = 0

    for r in range(game.board.rows):
        for c in range(game.board.cols):
            piece = game.board.get_piece(r, c)
            distance = math.sqrt((r - center_r) ** 2 + (c - center_c) ** 2)
            value = 10 / (1 + distance)  # Im bliżej centrum, tym większa wartość

            if piece == ai_player.color:
                score += value
            elif piece == game.opponent[ai_player.color]:
                score -= value

    return score


@HeuristicFactory.register(HeuristicType.HYBRID)
def aggressiveness_heuristic(ai_player: AIPlayer, game) -> float:
    """Promuje pozycje z większą liczbą możliwych ataków"""
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    score = 0

    for r in range(game.board.rows):
        for c in range(game.board.cols):
            piece = game.board.get_piece(r, c)
            if piece == ai_player.color:
                # Zlicz możliwości ataku
                attacks = sum(1 for dr, dc in directions
                              if (0 <= r + dr < game.board.rows and
                                  0 <= c + dc < game.board.cols and
                                  game.board.get_piece(r + dr, c + dc) == game.opponent[ai_player.color]))

            elif piece == game.opponent[ai_player.color]:
                # Analogicznie dla przeciwnika
                attacks = sum(1 for dr, dc in directions
                              if (0 <= r + dr < game.board.rows and
                                  0 <= c + dc < game.board.cols and
                                  game.board.get_piece(r + dr, c + dc) == ai_player.color))
                score -= attacks * 2

    return score


@HeuristicFactory.register(HeuristicType.DEFENSIVE)
def defensive_heuristic(ai_player: AIPlayer, game) -> float:
    """Kara za pionki narażone na zbicie"""
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    score = 0

    for r in range(game.board.rows):
        for c in range(game.board.cols):
            piece = game.board.get_piece(r, c)
            if piece == ai_player.color:
                # Sprawdź czy pionek jest zagrożony
                threatened = any(
                    game.board.get_piece(r + dr, c + dc) == game.opponent[ai_player.color]
                    for dr, dc in directions
                    if 0 <= r + dr < game.board.rows and 0 <= c + dc < game.board.cols
                )
                if threatened:
                    score -= 3

            elif piece == game.opponent[ai_player.color]:
                # Nagradzaj zagrożenie pionków przeciwnika
                threatened = any(
                    game.board.get_piece(r + dr, c + dc) == ai_player.color
                    for dr, dc in directions
                    if 0 <= r + dr < game.board.rows and 0 <= c + dc < game.board.cols
                )
                if threatened:
                    score += 3

    return score


@HeuristicFactory.register(HeuristicType.GOLDEN_SPOTS)
def golden_spots_heuristic(ai_player: AIPlayer, game) -> float:
    """Specjalne wartości dla kluczowych pól (np. rogi, centrum)"""
    golden_spots = [
        (0, 0), (0, game.board.cols - 1),
        (game.board.rows - 1, 0), (game.board.rows - 1, game.board.cols - 1),
        (game.board.rows // 2, game.board.cols // 2)
    ]

    score = 0
    for r, c in golden_spots:
        piece = game.board.get_piece(r, c)
        if piece == ai_player.color:
            score += 5
        elif piece == game.opponent[ai_player.color]:
            score -= 5
    return score


@HeuristicFactory.register(HeuristicType.GROUP_MOBILITY)
def group_mobility_heuristic(ai_player: AIPlayer, game) -> float:
    """Nagradza tworzenie grup połączonych pionków"""
    visited = [[False for _ in range(game.board.cols)] for _ in range(game.board.rows)]
    total_value = 0

    for r in range(game.board.rows):
        for c in range(game.board.cols):
            if not visited[r][c] and game.board.get_piece(r, c) == ai_player.color:
                # Znajdź rozmiar grupy (BFS)
                group_size = 0
                queue = [(r, c)]
                visited[r][c] = True

                while queue:
                    cr, cc = queue.pop(0)
                    group_size += 1

                    for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                        nr, nc = cr + dr, cc + dc
                        if (0 <= nr < game.board.rows and
                                0 <= nc < game.board.cols and
                                not visited[nr][nc] and
                                game.board.get_piece(nr, nc) == ai_player.color):
                            visited[nr][nc] = True
                            queue.append((nr, nc))

                total_value += group_size ** 1.5  # Nieliniowa nagroda za większe grupy

    return total_value


@HeuristicFactory.register(HeuristicType.ADAPTIVE)
def adaptive_hybrid_heuristic(ai_player: AIPlayer, game) -> float:
    """Dynamicznie dostosowuje wagi w zależności od fazy gry"""
    total_pieces = sum(1 for row in game.board.board for cell in row if cell != '_')
    total_cells = game.board.rows * game.board.cols

    if total_pieces > total_cells * 0.7:  # Wczesna faza
        return (center_control_heuristic(ai_player, game) * 0.6 +
                mobility_heuristic(ai_player, game) * 0.4)

    elif total_pieces > total_cells * 0.4:  # Środkowa faza
        return (mobility_heuristic(ai_player, game) * 0.5 +
                aggressiveness_heuristic(ai_player, game) * 0.3 +
                piece_count_heuristic(ai_player, game) * 0.2)

    else:  # Końcowa faza
        return (piece_count_heuristic(ai_player, game) * 0.7 +
                defensive_heuristic(ai_player, game) * 0.3)

@HeuristicFactory.register(HeuristicType.OPPONENT_AWARE)
def opponent_aware_heuristic(ai_player: AIPlayer, game) -> float:
    """Dostosowuje się do stylu gry przeciwnika"""
    # Prosta analiza ostatnich ruchów przeciwnika
    aggressive_moves = sum(1 for m in ai_player.history[-5:] if m['aggressiveness'] > 0.7)

    if aggressive_moves >= 3:  # Przeciwnik agresywny
        return (defensive_heuristic(ai_player, game) * 0.6 +
                piece_count_heuristic(ai_player, game) * 0.4)
    else:  # Standardowa strategia
        return (mobility_heuristic(ai_player, game) * 0.5 +
                center_control_heuristic(ai_player, game) * 0.3 +
                golden_spots_heuristic(ai_player, game) * 0.2)


@HeuristicFactory.register(HeuristicType.LEARNING)
def learning_heuristic(ai_player: AIPlayer, game) -> float:
    """Dostosowuje wagi na podstawie doświadczenia"""
    weights = getattr(ai_player, 'learned_weights', {
        'mobility': 1.0,
        'center': 1.0,
        'aggression': 1.0,
        'defense': 1.0
    })

    score = (
            weights['mobility'] * mobility_heuristic(ai_player, game) +
            weights['center'] * center_control_heuristic(ai_player, game) +
            weights['aggression'] * aggressiveness_heuristic(ai_player, game) +
            weights['defense'] * defensive_heuristic(ai_player, game)
    )

    # Aktualizacja wag po zakończonej grze
    if game.is_game_over():
        result = 1 if game.current_player != ai_player.color else -1
        learning_rate = 0.01

        # Normalizacja i aktualizacja
        total = sum(abs(h(ai_player, game)) for h in [mobility_heuristic, center_control_heuristic,
                                                      aggressiveness_heuristic, defensive_heuristic])
        if total > 0:
            weights['mobility'] += learning_rate * result * mobility_heuristic(ai_player, game) / total
            weights['center'] += learning_rate * result * center_control_heuristic(ai_player, game) / total
            weights['aggression'] += learning_rate * result * aggressiveness_heuristic(ai_player, game) / total
            weights['defense'] += learning_rate * result * defensive_heuristic(ai_player, game) / total

        # Zabezpieczenie przed ujemnymi wagami
        for key in weights:
            weights[key] = max(0.1, weights[key])

        ai_player.learned_weights = weights

    return score


if __name__ == "__main__":
    main()