from copy import deepcopy
from uuid import uuid4


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
    def __init__(self, board, player_b, player_w):
        self.board = ClobberBoard(board)
        self.players = {'B': player_b, 'W': player_w}
        self.current_player = 'B'
        self.opponent = {'B': 'W', 'W': 'B'}
        self.game_id = str(uuid4())

    def copy(self):
        clone = ClobberGame(deepcopy(self.board.board), self.players['B'], self.players['W'])
        clone.current_player = self.current_player
        clone.game_id = self.game_id
        return clone

    def get_valid_moves(self):
        moves = []
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        for r in range(self.board.rows):
            for c in range(self.board.cols):
                if self.board.get_piece(r, c) == self.current_player:
                    for dr, dc in directions:
                        nr, nc = r + dr, c + dc
                        if (0 <= nr < self.board.rows and 0 <= nc < self.board.cols and
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
        return not self.get_valid_moves()

    def get_current_player(self):
        return self.players[self.current_player]

    def get_winner(self):
        return self.opponent[self.current_player] if self.is_game_over() else None
