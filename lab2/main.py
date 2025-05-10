import asyncio
import concurrent.futures
import json

import websockets

from clobber_game import ClobberGame
from game_controller import GameController
from heuristics import HeuristicType
from player_factory import PlayerFactory
from server import WebSocketGameServer
from setup_players import register_players


def simulate_game(h1, h2, initial_board):
    p1 = PlayerFactory.create('ai', color='B', depth=5, heuristic_type=h1)
    p2 = PlayerFactory.create('ai', color='W', depth=5, heuristic_type=h2)
    game = ClobberGame(initial_board, p1, p2)
    controller = GameController(game)
    winner = controller.play()
    return h1, h2, winner


def run_local_game():
    initial_board = [
        ['B', 'W', 'B', 'W', 'B'],
        ['W', 'B', 'W', 'B', 'W'],
        ['B', 'W', 'B', 'W', 'B'],
        ['W', 'B', 'W', 'B', 'W'],
        ['B', 'W', 'B', 'W', 'B'],
        ['W', 'B', 'W', 'B', 'W'],
    ]

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

    print("Choose player types:")
    print("1. Human vs Human")
    print("2. Human vs AI")
    print("3. AI vs AI")
    print("4. AI vs AI (different heuristics)")
    print("5. AI Tournament")

    choice = input("Enter your choice (1-5): ")

    if choice == '1':
        p1 = PlayerFactory.create('human', color='B')
        p2 = PlayerFactory.create('human', color='W')
        game = ClobberGame(initial_board, p1, p2)
        controller = GameController(game)
        controller.play()

    elif choice == '2':
        print("Choose AI heuristic:")
        for i, h in enumerate(heuristic_types, 1):
            print(f"{i}. {heuristic_descriptions[h]}")
        h_choice = int(input("Choice: ")) - 1
        heuristic = heuristic_types[h_choice]

        p1 = PlayerFactory.create('human', color='B')
        p2 = PlayerFactory.create('ai', color='W', heuristic_type=heuristic)
        game = ClobberGame(initial_board, p1, p2)
        controller = GameController(game)
        controller.play()

    elif choice == '3':
        p1 = PlayerFactory.create('ai', color='B')
        p2 = PlayerFactory.create('ai', color='W')
        game = ClobberGame(initial_board, p1, p2)
        controller = GameController(game)
        controller.play()

    elif choice == '4':
        print("Choose AI 1 heuristic:")
        for i, h in enumerate(heuristic_types, 1):
            print(f"{i}. {heuristic_descriptions[h]}")
        h1 = heuristic_types[int(input("AI 1: ")) - 1]

        print("Choose AI 2 heuristic:")
        for i, h in enumerate(heuristic_types, 1):
            print(f"{i}. {heuristic_descriptions[h]}")
        h2 = heuristic_types[int(input("AI 2: ")) - 1]

        p1 = PlayerFactory.create('ai', color='B', heuristic_type=h1)
        p2 = PlayerFactory.create('ai', color='W', heuristic_type=h2)
        game = ClobberGame(initial_board, p1, p2)
        controller = GameController(game)
        controller.play()

    elif choice == '5':
        results = {h: {'wins': 0, 'losses': 0} for h in heuristic_types}
        matches = [(h1, h2) for h1 in heuristic_types for h2 in heuristic_types if h1 != h2]
        print(f"Running tournament with {len(matches)} matches...")

        with concurrent.futures.ProcessPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(simulate_game, h1, h2, initial_board) for h1, h2 in matches]
            for i, future in enumerate(concurrent.futures.as_completed(futures), 1):
                h1, h2, winner = future.result()
                if winner == 'B':
                    results[h1]['wins'] += 1
                    results[h2]['losses'] += 1
                elif winner == 'W':
                    results[h2]['wins'] += 1
                    results[h1]['losses'] += 1
                print(f"{i}/{len(matches)} matches completed")

        print("\n=== TOURNAMENT RESULTS ===")
        for h, stats in results.items():
            print(f"{heuristic_descriptions[h]}: {stats['wins']} wins, {stats['losses']} losses")
        best = max(results.items(), key=lambda x: x[1]['wins'])[0]
        print(f"\n🏆 Best heuristic: {heuristic_descriptions[best]} ({results[best]['wins']} wins)")

    else:
        print("Invalid choice. Exiting.")


def run_websocket_server():
    async def start():
        server = WebSocketGameServer()
        ws_server = await websockets.serve(server.handle_client, "localhost", 8765)
        print("WebSocket server running at ws://localhost:8765")
        await ws_server.wait_closed()

    asyncio.run(start())


def run_websocket_client():
    async def connect():
        uri = "ws://localhost:8765"
        async with websockets.connect(uri) as websocket:
            print("Connected to server")

            print("Choose your player type:")
            print("1. Human (play manually)")
            print("2. AI (automated play)")

            player_choice = input("Enter your choice (1-2): ")

            if player_choice == '1':
                player_type = "ws_human"
                heuristic_type = None
                depth = None
            else:
                player_type = "ai"

                print("Choose AI heuristic:")
                heuristics = list(HeuristicType)
                for i, h in enumerate(heuristics, 1):
                    print(f"{i}. {h.name}")
                h_choice = int(input("Choice: ")) - 1
                heuristic_type = heuristics[h_choice]

                depth = int(input("Enter AI search depth (e.g. 3): "))

            # Send registration message to server
            await websocket.send(json.dumps({
                "type": "register",
                "player_type": "ws_human",
                "opponent_type": "human"
            }))

            print("Waiting for game to start...")
            while True:
                msg = await websocket.recv()
                data = json.loads(msg)
                print("Message:", data)

                if data["type"] == "request_move":
                    print("Current board:")
                    for row in data["board"]:
                        print(" ".join(row))
                    print(f"Valid moves: {data['valid_moves']}")

                    if player_type == "ws_human":
                        move_input = input("Enter move (r1 c1 r2 c2): ")
                        r1, c1, r2, c2 = map(int, move_input.strip().split())
                        await websocket.send(json.dumps({
                            "type": "make_move",
                            "move": [(r1, c1), (r2, c2)]
                        }))
                    elif player_type == "ai":
                        from clobber_game import ClobberGame  # Import jeśli potrzebny
                        temp_game = ClobberGame(data["board"], current_player=data['current_player'])

                        # Create AI player with correct color
                        ai_player = PlayerFactory.create('ai',
                                                         color=data['current_player'],
                                                         depth=depth,
                                                         heuristic_type=heuristic_type)

                        move = ai_player.get_move(temp_game)
                        if move is None:
                            print("AI resigned!")
                            break
                        print(f"AI move: {move}")
                        r1, c1, r2, c2 = move[0][0], move[0][1], move[1][0], move[1][1]

                        await websocket.send(json.dumps({
                            "type": "make_move",
                            "move": [(r1, c1), (r2, c2)]
                        }))

                elif data["type"] == "move_made":
                    print(f"Move: {data['move']}")
                    for row in data["board"]:
                        print(" ".join(row))

                elif data["type"] == "game_over":
                    print(f"Game over! Winner: {data['winner']}")
                    break

                elif data["type"] == "waiting_for_opponent":
                    print("Waiting for opponent...")

                elif data["type"] == "game_started":
                    print(f"Game started! You are {data['color']}")
                    for row in data["board"]:
                        print(" ".join(row))

    asyncio.run(connect())


def main():
    register_players()

    print("Choose mode:")
    print("1. Local game")
    print("2. WebSocket server")
    print("3. Connect to WebSocket server")
    choice = input("Enter your choice (1-3): ")

    if choice == '1':
        run_local_game()
    elif choice == '2':
        run_websocket_server()
    elif choice == '3':
        run_websocket_client()
    else:
        print("Invalid choice.")


if __name__ == "__main__":
    main()
