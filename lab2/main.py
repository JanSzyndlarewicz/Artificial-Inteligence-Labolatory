from copy import deepcopy


class Clobber:
    def __init__(self, board, depth=3, player_vs_ai=True, player_color='B'):
        self.board = board
        self.rows = len(board)
        self.cols = len(board[0])
        self.current_player = 'B'
        self.opponent = {'B': 'W', 'W': 'B'}
        self.depth = depth
        self.player_vs_ai = player_vs_ai
        self.player_color = player_color

    def get_valid_moves(self):
        moves = []
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        for r in range(self.rows):
            for c in range(self.cols):
                if self.board[r][c] == self.current_player:
                    for dr, dc in directions:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < self.rows and 0 <= nc < self.cols and self.board[nr][nc] == self.opponent[
                            self.current_player]:
                            moves.append(((r, c), (nr, nc)))
        return moves

    def make_move(self, move):
        (r1, c1), (r2, c2) = move
        new_board = deepcopy(self.board)
        new_board[r2][c2] = self.current_player
        new_board[r1][c1] = '_'
        return new_board

    def switch_player(self):
        self.current_player = self.opponent[self.current_player]

    def is_game_over(self):
        return len(self.get_valid_moves()) == 0

    def print_board(self):
        for row in self.board:
            print(" ".join(row))
        print()

    def evaluate(self):
        return len(self.get_valid_moves())

    def minimax(self, depth, alpha, beta, maximizing):
        if depth == 0 or self.is_game_over():
            return self.evaluate()

        moves = self.get_valid_moves()
        if maximizing:
            max_eval = float('-inf')
            for move in moves:
                new_board = self.make_move(move)
                new_game = Clobber(new_board, self.depth, self.player_vs_ai, self.player_color)
                new_game.switch_player()
                eval_ = new_game.minimax(depth - 1, alpha, beta, False)
                max_eval = max(max_eval, eval_)
                alpha = max(alpha, eval_)
                if beta <= alpha:
                    break
            return max_eval
        else:
            min_eval = float('inf')
            for move in moves:
                new_board = self.make_move(move)
                new_game = Clobber(new_board, self.depth, self.player_vs_ai, self.player_color)
                new_game.switch_player()
                eval_ = new_game.minimax(depth - 1, alpha, beta, True)
                min_eval = min(min_eval, eval_)
                beta = min(beta, eval_)
                if beta <= alpha:
                    break
            return min_eval

    def get_best_move(self):
        best_move = None
        best_value = float('-inf')
        alpha, beta = float('-inf'), float('inf')
        for move in self.get_valid_moves():
            new_board = self.make_move(move)
            new_game = Clobber(new_board, self.depth, self.player_vs_ai, self.player_color)
            new_game.switch_player()
            move_value = new_game.minimax(self.depth - 1, alpha, beta, False)
            if move_value > best_value:
                best_value = move_value
                best_move = move
        return best_move

    def play_game(self):
        while not self.is_game_over():
            self.print_board()
            if self.player_vs_ai and self.current_player == self.player_color:
                move = self.get_human_move()

            else:
                move = self.get_best_move()

            if not move:
                break
            self.board = self.make_move(move)
            self.switch_player()

        self.print_board()
        print(f"Game over. Winner: {self.opponent[self.current_player]}")

    def get_human_move(self):
        valid_moves = self.get_valid_moves()
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
    game = Clobber(default_board, depth=3, player_vs_ai=player_vs_ai, player_color=color)
    game.play_game()

