import math
from enum import Enum, auto
from typing import Callable, Dict

from players import AIPlayer


class HeuristicType(Enum):
    PIECE_COUNT = auto()
    CENTER_CONTROL = auto()
    AGGRESSIVE = auto()
    DEFENSIVE = auto()
    GOLDEN_SPOTS = auto()
    GROUP_MOBILITY = auto()
    ADAPTIVE = auto()
    OPPONENT_AWARE = auto()
    LEARNING = auto()
    STRATEGIC = auto()


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
        return cls._heuristics.get(heuristic_type, cls._heuristics[HeuristicType.PIECE_COUNT])


def default_heuristic(ai_player: AIPlayer, game) -> float:
    """Domyślna heurystyka oparta na mobilności"""
    return piece_count_heuristic(ai_player, game)




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
            value = 10 / (1 + distance)

            if piece == ai_player.color:
                score += value
            elif piece == game.opponent[ai_player.color]:
                score -= value

    return score


@HeuristicFactory.register(HeuristicType.AGGRESSIVE)
def aggressiveness_heuristic(ai_player: AIPlayer, game) -> float:
    """Promuje pozycje z większą liczbą możliwych ataków"""
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    score = 0

    for r in range(game.board.rows):
        for c in range(game.board.cols):
            piece = game.board.get_piece(r, c)
            if piece == ai_player.color:
                # Zlicz możliwości ataku
                attacks = sum(
                    1
                    for dr, dc in directions
                    if (
                        0 <= r + dr < game.board.rows
                        and 0 <= c + dc < game.board.cols
                        and game.board.get_piece(r + dr, c + dc) == game.opponent[ai_player.color]
                    )
                )

            elif piece == game.opponent[ai_player.color]:
                # Analogicznie dla przeciwnika
                attacks = sum(
                    1
                    for dr, dc in directions
                    if (
                        0 <= r + dr < game.board.rows
                        and 0 <= c + dc < game.board.cols
                        and game.board.get_piece(r + dr, c + dc) == ai_player.color
                    )
                )
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
        (0, 0),
        (0, game.board.cols - 1),
        (game.board.rows - 1, 0),
        (game.board.rows - 1, game.board.cols - 1),
        (game.board.rows // 2, game.board.cols // 2),
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
                group_size = 0
                queue = [(r, c)]
                visited[r][c] = True

                while queue:
                    cr, cc = queue.pop(0)
                    group_size += 1

                    for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                        nr, nc = cr + dr, cc + dc
                        if (
                            0 <= nr < game.board.rows
                            and 0 <= nc < game.board.cols
                            and not visited[nr][nc]
                            and game.board.get_piece(nr, nc) == ai_player.color
                        ):
                            visited[nr][nc] = True
                            queue.append((nr, nc))

                total_value += group_size**1.5

    return total_value


@HeuristicFactory.register(HeuristicType.ADAPTIVE)
def adaptive_hybrid_heuristic(ai_player: AIPlayer, game) -> float:
    """Dynamicznie dostosowuje wagi w zależności od fazy gry"""
    total_pieces = sum(1 for row in game.board.board for cell in row if cell != "_")
    total_cells = game.board.rows * game.board.cols

    if total_pieces > total_cells * 0.7:  # Wczesna faza
        return center_control_heuristic(ai_player, game) * 0.6 + piece_count_heuristic(ai_player, game) * 0.4

    elif total_pieces > total_cells * 0.4:  # Środkowa faza
        return (
            piece_count_heuristic(ai_player, game) * 0.5
            + aggressiveness_heuristic(ai_player, game) * 0.3
            + piece_count_heuristic(ai_player, game) * 0.2
        )

    else:
        return piece_count_heuristic(ai_player, game) * 0.7 + defensive_heuristic(ai_player, game) * 0.3


@HeuristicFactory.register(HeuristicType.OPPONENT_AWARE)
def opponent_aware_heuristic(ai_player: AIPlayer, game) -> float:
    """Dostosowuje się do stylu gry przeciwnika"""
    # Prosta analiza ostatnich ruchów przeciwnika
    aggressive_moves = sum(1 for m in ai_player.history[-5:] if m["aggressiveness"] > 0.7)

    if aggressive_moves >= 3:  # Przeciwnik agresywny
        return defensive_heuristic(ai_player, game) * 0.6 + piece_count_heuristic(ai_player, game) * 0.4
    else:  # Standardowa strategia
        return (
            piece_count_heuristic(ai_player, game) * 0.5
            + center_control_heuristic(ai_player, game) * 0.3
            + golden_spots_heuristic(ai_player, game) * 0.2
        )


@HeuristicFactory.register(HeuristicType.LEARNING)
def learning_heuristic(ai_player: AIPlayer, game) -> float:
    """Dostosowuje wagi na podstawie doświadczenia"""
    weights = getattr(ai_player, "learned_weights", {"mobility": 1.0, "center": 1.0, "aggression": 1.0, "defense": 1.0})

    score = (
        weights["mobility"] * piece_count_heuristic(ai_player, game)
        + weights["center"] * center_control_heuristic(ai_player, game)
        + weights["aggression"] * aggressiveness_heuristic(ai_player, game)
        + weights["defense"] * defensive_heuristic(ai_player, game)
    )

    # Aktualizacja wag po zakończonej grze
    if game.is_game_over():
        result = 1 if game.current_player != ai_player.color else -1
        learning_rate = 0.01

        # Normalizacja i aktualizacja
        total = sum(
            abs(h(ai_player, game))
            for h in [piece_count_heuristic, center_control_heuristic, aggressiveness_heuristic, defensive_heuristic]
        )
        if total > 0:
            weights["mobility"] += learning_rate * result * piece_count_heuristic(ai_player, game) / total
            weights["center"] += learning_rate * result * center_control_heuristic(ai_player, game) / total
            weights["aggression"] += learning_rate * result * aggressiveness_heuristic(ai_player, game) / total
            weights["defense"] += learning_rate * result * defensive_heuristic(ai_player, game) / total

        # Zabezpieczenie przed ujemnymi wagami
        for key in weights:
            weights[key] = max(0.1, weights[key])

        ai_player.learned_weights = weights

    return score


@HeuristicFactory.register(HeuristicType.STRATEGIC)
def strategic_heuristic(ai_player: AIPlayer, game) -> float:
    """
    Comprehensive heuristic incorporating multiple advanced strategies:
    1. Symmetry awareness (for even-sized boards)
    2. Parity awareness (odd/even move count consideration)
    3. Divide and conquer (subgame analysis)
    4. Center control
    5. Avoidance of isolated stones
    """
    if game.is_game_over():
        return math.inf if game.current_player != ai_player.color else -math.inf

    score = 0.0

    # 1. Symmetry awareness (only for even-sized boards)
    if game.board.rows % 2 == 0 and game.board.cols % 2 == 0:
        symmetry_score = 0
        for r in range(game.board.rows // 2):
            for c in range(game.board.cols // 2):
                # Check if pieces are symmetric
                opposite_r = game.board.rows - 1 - r
                opposite_c = game.board.cols - 1 - c
                if game.board.get_piece(r, c) == ai_player.color:
                    if game.board.get_piece(opposite_r, opposite_c) == ai_player.color:
                        symmetry_score += 1
                    elif game.board.get_piece(opposite_r, opposite_c) == game.opponent[ai_player.color]:
                        symmetry_score -= 2
        score += symmetry_score * 0.5

    # 2. Parity awareness (try to control the game length)
    total_pieces = sum(1 for row in game.board.board for cell in row if cell != "_")
    total_cells = game.board.rows * game.board.cols
    game_phase = total_pieces / total_cells

    # Early game: aim for even parity (if second player) or odd (if first)
    if game_phase < 0.5:
        if ai_player.color == "B":  # First player wants odd total moves
            parity_modifier = 1 if (total_cells - total_pieces) % 2 == 1 else -1
        else:  # Second player wants even total moves
            parity_modifier = 1 if (total_cells - total_pieces) % 2 == 0 else -1
        score += parity_modifier * 2

    # 3. Divide and conquer (analyze subgames/clusters)
    clusters = find_clusters(game.board, ai_player.color)
    opponent_clusters = find_clusters(game.board, game.opponent[ai_player.color])

    # Prefer having more small clusters than fewer large ones
    cluster_score = len(opponent_clusters) - len(clusters)
    # Penalize large opponent clusters
    cluster_score -= 0.1 * sum(len(c) for c in opponent_clusters)
    score += cluster_score * 1.5

    # 4. Center control (adapted from existing heuristic)
    center_r, center_c = game.board.rows / 2, game.board.cols / 2
    center_score = 0
    for r in range(game.board.rows):
        for c in range(game.board.cols):
            piece = game.board.get_piece(r, c)
            distance = math.sqrt((r - center_r) ** 2 + (c - center_c) ** 2)
            value = 10 / (1 + distance)
            if piece == ai_player.color:
                center_score += value
            elif piece == game.opponent[ai_player.color]:
                center_score -= value
    score += center_score * 0.8

    # 5. Avoid isolated stones
    isolated_penalty = 0
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    for r in range(game.board.rows):
        for c in range(game.board.cols):
            if game.board.get_piece(r, c) == ai_player.color:
                # Count adjacent opponent pieces
                adjacent_opponent = sum(
                    1 for dr, dc in directions
                    if (0 <= r + dr < game.board.rows and
                        0 <= c + dc < game.board.cols and
                        game.board.get_piece(r + dr, c + dc) == game.opponent[ai_player.color])
                )
                if adjacent_opponent == 0:
                    isolated_penalty -= 3
    score += isolated_penalty

    # Combine with mobility for dynamic play
    mobility = piece_count_heuristic(ai_player, game)
    score += mobility * 0.7

    return score


def find_clusters(board, color):
    """Helper function to find clusters of connected pieces"""
    clusters = []
    visited = [[False for _ in range(board.cols)] for _ in range(board.rows)]

    for r in range(board.rows):
        for c in range(board.cols):
            if not visited[r][c] and board.get_piece(r, c) == color:
                # BFS to find connected cluster
                cluster = []
                queue = [(r, c)]
                visited[r][c] = True

                while queue:
                    cr, cc = queue.pop(0)
                    cluster.append((cr, cc))

                    for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                        nr, nc = cr + dr, cc + dc
                        if (0 <= nr < board.rows and 0 <= nc < board.cols and
                                not visited[nr][nc] and board.get_piece(nr, nc) == color):
                            visited[nr][nc] = True
                            queue.append((nr, nc))

                clusters.append(cluster)

    return clusters