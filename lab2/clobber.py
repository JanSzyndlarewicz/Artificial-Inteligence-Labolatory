from copy import deepcopy


class Clobber:
    def __init__(self, board, player_color='B'):
        self.board = board
        self.rows = len(board)
        self.cols = len(board[0])
        self.current_player = 'B'
        self.opponent = {'B': 'W', 'W': 'B'}
        self.player_color = player_color

    def get_valid_moves(self):
        moves = []
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        for r in range(self.rows):
            for c in range(self.cols):
                if self.board[r][c] == self.current_player:
                    for dr, dc in directions:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < self.rows and 0 <= nc < self.cols and self.board[nr][nc] == self.opponent[self.current_player]:
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
