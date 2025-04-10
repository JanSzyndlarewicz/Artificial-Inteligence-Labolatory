from copy import deepcopy
from abc import ABC, abstractmethod


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


class AIPlayer(Player):
    def __init__(self, color, depth=3):
        super().__init__(color)
        self.depth = depth

    def get_move(self, game):
        print(f"AI {self.color} is thinking...")
        best_move = None
        best_value = float('-inf')
        alpha, beta = float('-inf'), float('inf')

        for move in game.get_valid_moves():
            new_game = game.copy()
            new_game.make_move(move)
            move_value = self.minimax(new_game, self.depth - 1, alpha, beta, False)

            if move_value > best_value:
                best_value = move_value
                best_move = move

            alpha = max(alpha, best_value)

        return best_move

    def minimax(self, game, depth, alpha, beta, maximizing):
        if depth == 0 or game.is_game_over():
            return self.evaluate(game)

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

    def evaluate(self, game):
        # Simple evaluation function - count available moves
        return len(game.get_valid_moves())


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
        for row in self.board:
            print(" ".join(row))
        print()


class ClobberGame:
    def __init__(self, board, player1, player2):
        self.board = ClobberBoard(board)
        self.players = {'B': player1, 'W': player2}
        self.current_player = 'B'
        self.opponent = {'B': 'W', 'W': 'B'}

    def copy(self):
        new_game = ClobberGame(deepcopy(self.board.board),
                               self.players['B'],
                               self.players['W'])
        new_game.current_player = self.current_player
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

    def play(self):
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


def create_player(player_type, color, depth=5):
    if player_type.lower() == 'human':
        return HumanPlayer(color)
    elif player_type.lower() == 'ai':
        return AIPlayer(color, depth)
    else:
        raise ValueError("Invalid player type. Choose 'human' or 'ai'.")


def main():
    # Example board
    initial_board = [
        ['B', 'W', 'B', 'W', 'B'],
        ['W', 'B', 'W', 'B', 'W'],
        ['B', 'W', 'B', 'W', 'B'],
        ['W', 'B', 'W', 'B', 'W'],
        ['B', 'W', 'B', 'W', 'B'],
        ['W', 'B', 'W', 'B', 'W'],
    ]

    print("Choose player types:")
    print("1. Human vs Human")
    print("2. Human vs AI")
    print("3. AI vs AI")

    choice = input("Enter your choice (1-3): ")

    if choice == '1':
        player1 = create_player('human', 'B')
        player2 = create_player('human', 'W')
    elif choice == '2':
        player1 = create_player('human', 'B')
        player2 = create_player('ai', 'W')
    elif choice == '3':
        player1 = create_player('ai', 'B')
        player2 = create_player('ai', 'W')
    else:
        print("Invalid choice. Defaulting to Human vs AI.")
        player1 = create_player('human', 'B')
        player2 = create_player('ai', 'W')

    game = ClobberGame(initial_board, player1, player2)
    game.play()


if __name__ == "__main__":
    main()