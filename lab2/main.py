import asyncio
import concurrent.futures
import json
from typing import Optional, Tuple

import websockets
from clobber_game import ClobberGame
from game_controller import GameController
from heuristics import HeuristicType
from message_types import MessageType
from player_factory import PlayerFactory
from server import WebSocketGameServer
from setup_players import register_players

initial_board = [
    ["B", "W", "B", "W", "B"],
    ["W", "B", "W", "B", "W"],
    ["B", "W", "B", "W", "B"],
    ["W", "B", "W", "B", "W"],
    ["B", "W", "B", "W", "B"],
    ["W", "B", "W", "B", "W"],
]

HEURISTIC_DESCRIPTIONS = {
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
    HeuristicType.HYBRID: "Hybrid",
}


class GameSetup:
    @staticmethod
    def print_heuristic_options() -> None:
        print("Choose heuristic:")
        for i, h in enumerate(HeuristicType, 1):
            print(f"{i}. {HEURISTIC_DESCRIPTIONS[h]}")

    @staticmethod
    def select_heuristic(prompt: str = "Choice: ") -> HeuristicType:
        GameSetup.print_heuristic_options()
        h_choice = int(input(prompt)) - 1
        return list(HeuristicType)[h_choice]

    @staticmethod
    def create_players(
        player1_type: str,
        player2_type: str,
        p1_heuristic: Optional[HeuristicType] = None,
        p2_heuristic: Optional[HeuristicType] = None,
        depth1: int = 3,
        depth2: int = 3,
    ) -> Tuple:
        p1 = PlayerFactory.create(
            player1_type, color="B", heuristic_type=p1_heuristic, depth=depth1 if player1_type == "ai" else None
        )
        p2 = PlayerFactory.create(
            player2_type, color="W", heuristic_type=p2_heuristic, depth=depth2 if player2_type == "ai" else None
        )
        return p1, p2

    @staticmethod
    def start_game(p1, p2) -> str:
        game = ClobberGame(initial_board, p1, p2)
        controller = GameController(game)
        return controller.play()

    @staticmethod
    def print_board(board):
        for row in board:
            print(" ".join(row))


def simulate_game(h1: HeuristicType, h2: HeuristicType) -> Tuple:
    p1, p2 = GameSetup.create_players("ai", "ai", h1, h2)
    winner = GameSetup.start_game(p1, p2)
    return h1, h2, winner


def run_local_game():
    print("Choose player types:")
    print("1. Human vs Human")
    print("2. Human vs AI")
    print("3. AI vs AI")
    print("4. AI vs AI (different heuristics)")
    print("5. AI Tournament")

    choice = input("Enter your choice (1-5): ")

    if choice == "1":
        p1, p2 = GameSetup.create_players("human", "human")
        GameSetup.start_game(p1, p2)

    elif choice == "2":
        heuristic = GameSetup.select_heuristic()
        p1, p2 = GameSetup.create_players("human", "ai", None, heuristic)
        GameSetup.start_game(p1, p2)

    elif choice == "3":
        p1, p2 = GameSetup.create_players("ai", "ai")
        GameSetup.start_game(p1, p2)

    elif choice == "4":
        print("Choose AI 1 heuristic:")
        h1 = GameSetup.select_heuristic("AI 1: ")
        print("Select depth for AI 1:")
        depth1 = int(input("Depth (e.g. 3): "))

        print("Choose AI 2 heuristic:")
        h2 = GameSetup.select_heuristic("AI 2: ")
        print("Select depth for AI 2:")
        depth2 = int(input("Depth (e.g. 3): "))

        p1, p2 = GameSetup.create_players("ai", "ai", h1, h2, depth1, depth2)
        GameSetup.start_game(p1, p2)

    elif choice == "5":
        results = {h: {"wins": 0, "losses": 0} for h in HeuristicType}
        matches = [(h1, h2) for h1 in HeuristicType for h2 in HeuristicType if h1 != h2]
        print(f"Running tournament with {len(matches)} matches...")

        with concurrent.futures.ProcessPoolExecutor(max_workers=18) as executor:
            futures = [executor.submit(simulate_game, h1, h2) for h1, h2 in matches]
            for i, future in enumerate(concurrent.futures.as_completed(futures), 1):
                h1, h2, winner = future.result()
                if winner == "B":
                    results[h1]["wins"] += 1
                    results[h2]["losses"] += 1
                elif winner == "W":
                    results[h2]["wins"] += 1
                    results[h1]["losses"] += 1
                print(f"{i}/{len(matches)} matches completed")

        print("\n=== TOURNAMENT RESULTS ===")
        for h, stats in results.items():
            print(f"{HEURISTIC_DESCRIPTIONS[h]}: {stats['wins']} wins, {stats['losses']} losses")
        best = max(results.items(), key=lambda x: x[1]["wins"])[0]
        print(f"\n🏆 Best heuristic: {HEURISTIC_DESCRIPTIONS[best]} ({results[best]['wins']} wins)")

    else:
        print("Invalid choice. Exiting.")


def run_websocket_server():
    async def start():
        server = WebSocketGameServer(board=initial_board)
        ws_server = await websockets.serve(server.handle_client, "localhost", 8765)
        print("WebSocket server running at ws://localhost:8765")
        await ws_server.wait_closed()

    asyncio.run(start())


async def run_websocket_client():
    async def connect():
        uri = "ws://localhost:8765"
        async with websockets.connect(uri) as websocket:
            print("Connected to server")

            # Uproszczony wybór typu gracza
            print("Choose your player type:")
            print("1. Human (play manually)")
            print("2. AI (automated play)")
            player_choice = input("Enter your choice (1-2): ")

            if player_choice == "1":
                player_type = "websocket"
                heuristic_type = None
                depth = None
            else:
                player_type = "ai"
                heuristic_type = GameSetup.select_heuristic()
                depth = int(input("Enter AI search depth (e.g. 3): "))

            # Rejestracja
            await websocket.send(
                json.dumps(
                    {
                        "type": MessageType.REGISTER,
                    }
                )
            )
            print("Waiting for game to start...")

            await handle_game_loop(websocket, player_type, heuristic_type, depth)

    async def handle_game_loop(websocket, player_type, heuristic_type, depth):
        while True:
            msg = await websocket.recv()
            data = json.loads(msg)
            print("Message:", data)

            if data["type"] == MessageType.REQUEST_MOVE:
                await handle_move_request(websocket, data, player_type, heuristic_type, depth)
            elif data["type"] == MessageType.MOVE_MADE:
                print(f"Move: {data['move']}")
                GameSetup.print_board(data["board"])
            elif data["type"] == MessageType.GAME_OVER:
                print(f"Game over! Winner: {data['winner']}")
                break
            elif data["type"] == MessageType.WAITING:
                print("Waiting for opponent...")
            elif data["type"] == MessageType.GAME_STARTED:
                print(f"Game started! You are {data['color']}")
                GameSetup.print_board(data["board"])

    async def handle_move_request(websocket, data, player_type, heuristic_type, depth):
        print("Current board:")
        GameSetup.print_board(data["board"])
        print(f"Valid moves: {data['valid_moves']}")

        if player_type == "websocket":
            move_input = input("Enter move (r1 c1 r2 c2): ")
            r1, c1, r2, c2 = map(int, move_input.strip().split())
            await send_move(websocket, [(r1, c1), (r2, c2)])
        else:
            await handle_ai_move(websocket, data, heuristic_type, depth)

    async def handle_ai_move(websocket, data, heuristic_type, depth):
        temp_game = ClobberGame(data["board"], current_player=data["current_player"])
        ai_player = PlayerFactory.create("ai", color=data["current_player"], depth=depth, heuristic_type=heuristic_type)

        move = ai_player.get_move(temp_game)
        if move is None:
            print("AI resigned!")
            return

        print(f"AI move: {move}")
        await send_move(websocket, move)

    async def send_move(websocket, move):
        r1, c1, r2, c2 = move[0][0], move[0][1], move[1][0], move[1][1]
        await websocket.send(json.dumps({"type": MessageType.MAKE_MOVE, "move": [(r1, c1), (r2, c2)]}))

    await connect()


def main():
    register_players()

    print("Choose mode:")
    print("1. Local game")
    print("2. WebSocket server")
    print("3. Connect to WebSocket server")
    choice = input("Enter your choice (1-3): ")

    if choice == "1":
        run_local_game()
    elif choice == "2":
        run_websocket_server()
    elif choice == "3":
        run_websocket_client()
    else:
        print("Invalid choice.")


if __name__ == "__main__":
    main()
