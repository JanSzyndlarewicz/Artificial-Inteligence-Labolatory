from bot_player import BotPlayer
from clobber import Clobber


class ClobberGame:
    def __init__(self, board, player_vs_ai=True, player_color='B'):
        self.game = Clobber(board, player_color)
        self.bot = BotPlayer(self.game)
        self.player_vs_ai = player_vs_ai

    def play_game(self):
        while not self.game.is_game_over():
            self.game.print_board()
            if self.player_vs_ai and self.game.current_player == self.game.player_color:
                move = self.get_human_move()
            else:
                move = self.bot.get_best_move()

            if not move:
                break
            self.game.board = self.game.make_move(move)
            self.game.switch_player()

        self.game.print_board()
        print(f"Game over. Winner: {self.game.opponent[self.game.current_player]}")

    def get_human_move(self):
        valid_moves = self.game.get_valid_moves()
        while True:
            try:
                user_input = input("Enter move (row1 col1 row2 col2): ").strip()
                if user_input.lower() == "quit":
                    return None
                r1, c1, r2, c2 = map(int, user_input.split())
                move = ((r1, c1), (r2, c2))
                if move in valid_moves:
                    return move
            except ValueError:
                pass
            print("Invalid move, try again.")
