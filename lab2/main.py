from copy import deepcopy
from abc import ABC, abstractmethod
import math
import concurrent.futures



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
    def __init__(self, color, depth=3, heuristic_type='mobility'):
        super().__init__(color)
        self.depth = depth
        self.heuristic_type = heuristic_type
        # Dane dla heurystyk adaptacyjnych
        self.opponent_style = None  # 'aggressive'/'defensive'/'balanced'
        self.history = []
        self.learned_weights = {
            'center': 1.0,
            'mobility': 1.0,
            'aggression': 1.0,
            'defense': 1.0
        }

    def get_move(self, game):
        print(f"AI {self.color} is thinking with {self.heuristic_type} heuristic...")
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
        """Evaluate the game state based on the selected heuristic"""
        if self.heuristic_type == 'mobility':
            return self._mobility_heuristic(game)
        elif self.heuristic_type == 'piece_count':
            return self._piece_count_heuristic(game)
        elif self.heuristic_type == 'positional':
            return self._positional_heuristic(game)
        elif self.heuristic_type == 'center_control':
            return self._center_control_heuristic(game)
        elif self.heuristic_type == 'aggressiveness':
            return self._aggressiveness_heuristic(game)
        elif self.heuristic_type == 'defensive':
            return self._defensive_heuristic(game)
        elif self.heuristic_type == 'golden_spots':
            return self._golden_spots_heuristic(game)
        elif self.heuristic_type == 'group_mobility':
            return self._group_mobility_heuristic(game)
        else:
            return self._mobility_heuristic(game)  # default

    def _mobility_heuristic(self, game):
        """Heuristic 1: Mobility - difference in available moves"""
        if game.is_game_over():
            return math.inf if game.current_player != self.color else -math.inf

        current_moves = len(game.get_valid_moves())
        game.switch_player()
        opponent_moves = len(game.get_valid_moves())
        game.switch_player()

        # Maximize the difference between our moves and opponent's moves
        return current_moves - opponent_moves

    def _piece_count_heuristic(self, game):
        """Heuristic 2: Piece Count - difference in remaining pieces"""
        if game.is_game_over():
            return math.inf if game.current_player != self.color else -math.inf

        current_pieces = 0
        opponent_pieces = 0

        for r in range(game.board.rows):
            for c in range(game.board.cols):
                piece = game.board.get_piece(r, c)
                if piece == self.color:
                    current_pieces += 1
                elif piece == game.opponent[self.color]:
                    opponent_pieces += 1

        return current_pieces - opponent_pieces

    def _positional_heuristic(self, game):
        """Heuristic 3: Positional - evaluate board control and piece positions"""
        if game.is_game_over():
            return math.inf if game.current_player != self.color else -math.inf

        score = 0
        center_weight = 2  # Higher value for center positions
        edge_weight = 1  # Standard value for edge positions

        for r in range(game.board.rows):
            for c in range(game.board.cols):
                piece = game.board.get_piece(r, c)
                if piece == self.color:
                    # Center control is more valuable
                    distance_to_center = math.sqrt((r - game.board.rows / 2) ** 2 + (c - game.board.cols / 2) ** 2)
                    position_value = 1 / (1 + distance_to_center)
                    score += position_value * 10

                    # Bonus for pieces that can attack multiple directions
                    attack_directions = 0
                    for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                        nr, nc = r + dr, c + dc
                        if (0 <= nr < game.board.rows and 0 <= nc < game.board.cols and
                                game.board.get_piece(nr, nc) == game.opponent[self.color]):
                            attack_directions += 1
                    score += attack_directions * 0.5

                elif piece == game.opponent[self.color]:
                    # Same calculations but subtract for opponent
                    distance_to_center = math.sqrt((r - game.board.rows / 2) ** 2 + (c - game.board.cols / 2) ** 2)
                    position_value = 1 / (1 + distance_to_center)
                    score -= position_value * 10

                    attack_directions = 0
                    for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                        nr, nc = r + dr, c + dc
                        if (0 <= nr < game.board.rows and 0 <= nc < game.board.cols and
                                game.board.get_piece(nr, nc) == self.color):
                            attack_directions += 1
                    score -= attack_directions * 0.5

        return score

    def _center_control_heuristic(self, game):
        if game.is_game_over():
            return math.inf if game.current_player != self.color else -math.inf

        center_row, center_col = game.board.rows / 2, game.board.cols / 2
        score = 0

        for r in range(game.board.rows):
            for c in range(game.board.cols):
                piece = game.board.get_piece(r, c)
                distance_to_center = math.sqrt((r - center_row) ** 2 + (c - center_col) ** 2)
                position_value = 1 / (1 + distance_to_center)  # Im bliżej centrum, tym wyższa wartość

                if piece == self.color:
                    score += position_value * 10
                elif piece == game.opponent[self.color]:
                    score -= position_value * 10

        return score

    def _aggressiveness_heuristic(self, game):
        if game.is_game_over():
            return math.inf if game.current_player != self.color else -math.inf

        score = 0
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]

        for r in range(game.board.rows):
            for c in range(game.board.cols):
                piece = game.board.get_piece(r, c)
                if piece == self.color:
                    # Zlicz możliwe ataki dla danego pionka
                    attack_options = 0
                    for dr, dc in directions:
                        nr, nc = r + dr, c + dc
                        if (0 <= nr < game.board.rows and 0 <= nc < game.board.cols and
                                game.board.get_piece(nr, nc) == game.opponent[self.color]):
                            attack_options += 1
                    score += attack_options * 2  # Większa wartość za więcej opcji ataku

                elif piece == game.opponent[self.color]:
                    # Analogicznie dla przeciwnika (odejmujemy)
                    attack_options = 0
                    for dr, dc in directions:
                        nr, nc = r + dr, c + dc
                        if (0 <= nr < game.board.rows and 0 <= nc < game.board.cols and
                                game.board.get_piece(nr, nc) == self.color):
                            attack_options += 1
                    score -= attack_options * 2

        return score

    def _defensive_heuristic(self, game):
        if game.is_game_over():
            return math.inf if game.current_player != self.color else -math.inf

        score = 0
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]

        for r in range(game.board.rows):
            for c in range(game.board.cols):
                piece = game.board.get_piece(r, c)
                if piece == self.color:
                    # Sprawdź, czy pionek jest zagrożony (czy przeciwnik może go zbić)
                    is_threatened = False
                    for dr, dc in directions:
                        nr, nc = r + dr, c + dc
                        if (0 <= nr < game.board.rows and 0 <= nc < game.board.cols and
                                game.board.get_piece(nr, nc) == game.opponent[self.color]):
                            is_threatened = True
                            break
                    if is_threatened:
                        score -= 3  # Kara za zagrożone pionki

                elif piece == game.opponent[self.color]:
                    # Analogicznie dla przeciwnika (nagradzaj, jeśli jego pionki są zagrożone)
                    is_threatened = False
                    for dr, dc in directions:
                        nr, nc = r + dr, c + dc
                        if (0 <= nr < game.board.rows and 0 <= nc < game.board.cols and
                                game.board.get_piece(nr, nc) == self.color):
                            is_threatened = True
                            break
                    if is_threatened:
                        score += 3

        return score

    def _golden_spots_heuristic(self, game):
        if game.is_game_over():
            return math.inf if game.current_player != self.color else -math.inf

        # Przykładowe "złote punkty" - rogi i środki krawędzi
        golden_spots = [
            (0, 0), (0, game.board.cols - 1),
            (game.board.rows - 1, 0), (game.board.rows - 1, game.board.cols - 1),
            (0, game.board.cols // 2), (game.board.rows - 1, game.board.cols // 2),
            (game.board.rows // 2, 0), (game.board.rows // 2, game.board.cols - 1)
        ]

        score = 0
        for (r, c) in golden_spots:
            piece = game.board.get_piece(r, c)
            if piece == self.color:
                score += 5  # Wysoka nagroda za kontrolę złotego punktu
            elif piece == game.opponent[self.color]:
                score -= 5

        return score

    def _group_mobility_heuristic(self, game):
        if game.is_game_over():
            return math.inf if game.current_player != self.color else -math.inf

        visited = [[False for _ in range(game.board.cols)] for _ in range(game.board.rows)]
        total_group_value = 0

        for r in range(game.board.rows):
            for c in range(game.board.cols):
                if not visited[r][c] and game.board.get_piece(r, c) == self.color:
                    # Rozpocznij BFS, aby znaleźć rozmiar grupy
                    queue = [(r, c)]
                    visited[r][c] = True
                    group_size = 0

                    while queue:
                        current_r, current_c = queue.pop(0)
                        group_size += 1

                        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                            nr, nc = current_r + dr, current_c + dc
                            if (0 <= nr < game.board.rows and 0 <= nc < game.board.cols and
                                    not visited[nr][nc] and game.board.get_piece(nr, nc) == self.color):
                                visited[nr][nc] = True
                                queue.append((nr, nc))

                    # Nagroda za większe grupy (np. kwadratowo)
                    total_group_value += group_size ** 1.5

        return total_group_value

    def _adaptive_hybrid_heuristic(self, game):
        """Adaptacyjna hybrydowa heurystyka dostosowująca się do fazy gry"""
        phase = self._game_phase(game)

        if phase == 'early':
            # Wczesna faza - kontrola centrum i mobilność
            return 0.6 * self._center_control_heuristic(game) + \
                0.3 * self._mobility_heuristic(game) + \
                0.1 * self._aggressiveness_heuristic(game)

        elif phase == 'mid':
            # Środkowa faza - agresja i mobilność
            return 0.4 * self._aggressiveness_heuristic(game) + \
                0.4 * self._mobility_heuristic(game) + \
                0.2 * self._piece_count_heuristic(game)

        else:
            # Końcowa faza - liczba pionków i obrona
            return 0.7 * self._piece_count_heuristic(game) + \
                0.2 * self._defensive_heuristic(game) + \
                0.1 * self._golden_spots_heuristic(game)

    def _game_phase(self, game):
        """Określa fazę gry na podstawie liczby pozostałych pionków"""
        total_pieces = sum(1 for row in game.board.board for cell in row if cell != '_')
        total_cells = game.board.rows * game.board.cols

        if total_pieces > total_cells * 0.6:  # Więcej niż 60% pionków
            return 'early'
        elif total_pieces > total_cells * 0.3:  # 30%-60% pionków
            return 'mid'
        else:
            return 'late'

    def _style_aware_heuristic(self, game):
        """Heurystyka reagująca na styl gry przeciwnika"""
        self._update_opponent_style(game)

        if self.opponent_style == 'aggressive':
            # Przeciw agresywnemu przeciwnikowi - więcej obrony
            return 0.5 * self._defensive_heuristic(game) + \
                0.3 * self._aggressiveness_heuristic(game) + \
                0.2 * self._piece_count_heuristic(game)
        elif self.opponent_style == 'defensive':
            # Przeciw defensywnemu przeciwnikowi - kontrola centrum
            return 0.5 * self._center_control_heuristic(game) + \
                0.3 * self._mobility_heuristic(game) + \
                0.2 * self._golden_spots_heuristic(game)
        else:
            # Domyślna strategia zbalansowana
            return 0.4 * self._mobility_heuristic(game) + \
                0.3 * self._aggressiveness_heuristic(game) + \
                0.2 * self._center_control_heuristic(game) + \
                0.1 * self._defensive_heuristic(game)

    def _update_opponent_style(self, game):
        """Analizuje styl gry przeciwnika na podstawie historii ruchów"""
        if len(self.history) < 5:
            return

        # Oblicz średnią agresywność ostatnich ruchów
        avg_aggressiveness = sum(
            self._calculate_move_aggressiveness(move, game)
            for move in self.history[-5:]
        ) / 5

        if avg_aggressiveness > 0.7:
            self.opponent_style = 'aggressive'
        elif avg_aggressiveness < 0.3:
            self.opponent_style = 'defensive'
        else:
            self.opponent_style = 'balanced'

    def _calculate_move_aggressiveness(self, move, game):
        """Ocenia agresywność pojedynczego ruchu"""
        (r1, c1), (r2, c2) = move
        attack_options = 0

        # Sprawdź czy ruch tworzy nowe możliwości ataku
        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nr, nc = r2 + dr, c2 + dc
            if (0 <= nr < game.board.rows and 0 <= nc < game.board.cols and
                    game.board.get_piece(nr, nc) == self.color):
                attack_options += 1

        return min(attack_options / 4, 1.0)  # Normalizuj do zakresu 0-1

    def _learning_heuristic(self, game):
        """Heurystyka ucząca się na podstawie doświadczenia"""
        score = (
                self.learned_weights['center'] * self._center_control_heuristic(game) +
                self.learned_weights['mobility'] * self._mobility_heuristic(game) +
                self.learned_weights['aggression'] * self._aggressiveness_heuristic(game) +
                self.learned_weights['defense'] * self._defensive_heuristic(game)
        )

        # Po zakończonej grze aktualizuj wagi
        if game.is_game_over():
            self._update_weights(game)

        return score

    def _update_weights(self, game):
        """Aktualizuje wagi na podstawie wyniku gry"""
        result = 1 if game.current_player != self.color else -1
        learning_rate = 0.01

        # Normalizuj wartości heurystyk do zakresu 0-1
        max_possible = {
            'center': 10 * game.board.rows * game.board.cols,
            'mobility': game.board.rows * game.board.cols * 4,
            'aggression': game.board.rows * game.board.cols * 8,
            'defense': game.board.rows * game.board.cols * 3
        }

        # Aktualizuj wagi z uwzględnieniem normalizacji
        self.learned_weights['center'] += learning_rate * result * (
                self._center_control_heuristic(game) / max_possible['center']
        )
        self.learned_weights['mobility'] += learning_rate * result * (
                self._mobility_heuristic(game) / max_possible['mobility']
        )
        self.learned_weights['aggression'] += learning_rate * result * (
                self._aggressiveness_heuristic(game) / max_possible['aggression']
        )
        self.learned_weights['defense'] += learning_rate * result * (
                self._defensive_heuristic(game) / max_possible['defense']
        )

        # Upewnij się, że wagi pozostają dodatnie
        for key in self.learned_weights:
            self.learned_weights[key] = max(0.1, self.learned_weights[key])

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


def create_player(player_type, color, depth=3, heuristic='mobility'):
    if player_type.lower() == 'human':
        return HumanPlayer(color)
    elif player_type.lower() == 'ai':
        return AIPlayer(color, depth, heuristic)
    else:
        raise ValueError("Invalid player type. Choose 'human' or 'ai'.")

def simulate_game(h1, h2, initial_board):
    player1 = create_player('ai', 'B', 5, h1)
    player2 = create_player('ai', 'W', 5, h2)
    game = ClobberGame(initial_board, player1, player2)
    winner = game.play()
    return h1, h2, winner


def main():
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
    print("4. AI vs AI (different heuristics)")
    print("5. AI Tournament (each heuristic vs each)")

    choice = input("Enter your choice (1-5): ")

    heuristic_map = {
        'mobility': 'Mobility',
        'piece_count': 'Piece Count',
        'positional': 'Positional',
        'center_control': 'Center Control',
        'aggressiveness': 'Aggressiveness',
        'defensive': 'Defensive',
        'golden_spots': 'Golden Spots',
        'group_mobility': 'Group Mobility',
        'adaptive_hybrid': 'Adaptive Hybrid',
        'style_aware': 'Style Aware',
        'learning': 'Learning'
    }

    heuristics = list(heuristic_map.keys())

    if choice == '1':
        player1 = create_player('human', 'B')
        player2 = create_player('human', 'W')

        game = ClobberGame(initial_board, player1, player2)
        game.play()

    elif choice == '2':
        print("Choose AI heuristic:")
        for i, h in enumerate(heuristics, 1):
            print(f"{i}. {heuristic_map[h]}")
        heuristic_choice = input("Enter heuristic choice: ")
        heuristic = heuristics[int(heuristic_choice) - 1]

        player1 = create_player('human', 'B')
        player2 = create_player('ai', 'W', 3, heuristic)

        game = ClobberGame(initial_board, player1, player2)
        game.play()

    elif choice == '3':
        player1 = create_player('ai', 'B')
        player2 = create_player('ai', 'W')

        game = ClobberGame(initial_board, player1, player2)
        game.play()

    elif choice == '4':
        print("Choose AI 1 heuristic:")
        for i, h in enumerate(heuristics, 1):
            print(f"{i}. {heuristic_map[h]}")
        heuristic1 = heuristics[int(input("AI 1 choice: ")) - 1]

        print("Choose AI 2 heuristic:")
        for i, h in enumerate(heuristics, 1):
            print(f"{i}. {heuristic_map[h]}")
        heuristic2 = heuristics[int(input("AI 2 choice: ")) - 1]

        player1 = create_player('ai', 'B', 3, heuristic1)
        player2 = create_player('ai', 'W', 3, heuristic2)

        game = ClobberGame(initial_board, player1, player2)
        game.play()


    elif choice == '5':

        results = {h: {'wins': 0, 'losses': 0, 'draws': 0} for h in heuristics}

        matches = []

        for i, h1 in enumerate(heuristics):
            for j, h2 in enumerate(heuristics):
                if i == j:
                    continue
                matches.append((h1, h2))

        print(f"Running {len(matches)} games in parallel...\n")

        MAX_WORKERS = 16

        with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = [executor.submit(simulate_game, h1, h2, initial_board) for h1, h2 in matches]

            games_finished = 0
            for future in concurrent.futures.as_completed(futures):
                h1, h2, winner = future.result()

                if winner == 'B':
                    results[h1]['wins'] += 1
                    results[h2]['losses'] += 1
                elif winner == 'W':
                    results[h2]['wins'] += 1
                    results[h1]['losses'] += 1

                games_finished += 1
                print(f"Games finished: {games_finished}/{len(matches)}")

        print("\n=== TOURNAMENT RESULTS ===")

        for h, stats in results.items():
            print(f"{heuristic_map[h]}: {stats['wins']}W / {stats['losses']}L, score: {stats['wins'] / stats['losses']}")

        best = max(results.items(), key=lambda x: x[1]['wins'])[0]

        print(f"\nBest heuristic: {heuristic_map[best]}")

    else:
        print("Invalid choice. Defaulting to Human vs AI.")
        player1 = create_player('human', 'B')
        player2 = create_player('ai', 'W')

        game = ClobberGame(initial_board, player1, player2)
        game.play()


if __name__ == "__main__":
    main()