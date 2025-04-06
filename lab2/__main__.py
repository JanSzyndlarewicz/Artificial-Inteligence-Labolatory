from clobber_game import ClobberGame

if __name__ == "__main__":
    default_board = [
        ['B', 'W', 'B', 'W', 'B', 'W'],
        ['W', 'B', 'W', 'B', 'W', 'B'],
        ['B', 'W', 'B', 'W', 'B', 'W'],
        ['W', 'B', 'W', 'B', 'W', 'B'],
        ['B', 'W', 'B', 'W', 'B', 'W']
    ]
    mode = input("Play against AI? (yes/no): ").strip().lower()
    player_vs_ai = mode == "yes"

    if player_vs_ai:
        color = input("Choose your color (B/W): ").strip().upper()
        if color not in ['B', 'W']:
            color = 'B'
    else:
        color = 'B'

    game = ClobberGame(default_board, player_vs_ai=player_vs_ai, player_color=color)
    game.play_game()
