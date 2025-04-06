from clobber import Clobber


class BotPlayer:
    def __init__(self, game, depth=3):
        self.game = game
        self.depth = depth

    def minimax(self, depth, alpha, beta, maximizing):
        if depth == 0 or self.game.is_game_over():
            return self.game.evaluate()

        moves = self.game.get_valid_moves()
        if maximizing:
            max_eval = float('-inf')
            for move in moves:
                new_board = self.game.make_move(move)
                new_game = Clobber(new_board, self.game.player_color)
                new_game.switch_player()
                eval_ = BotPlayer(new_game, self.depth).minimax(depth - 1, alpha, beta, False)
                max_eval = max(max_eval, eval_)
                alpha = max(alpha, eval_)
                if beta <= alpha:
                    break
            return max_eval
        else:
            min_eval = float('inf')
            for move in moves:
                new_board = self.game.make_move(move)
                new_game = Clobber(new_board, self.game.player_color)
                new_game.switch_player()
                eval_ = BotPlayer(new_game, self.depth).minimax(depth - 1, alpha, beta, True)
                min_eval = min(min_eval, eval_)
                beta = min(beta, eval_)
                if beta <= alpha:
                    break
            return min_eval

    def get_best_move(self):
        best_move = None
        best_value = float('-inf')
        alpha, beta = float('-inf'), float('inf')
        for move in self.game.get_valid_moves():
            new_board = self.game.make_move(move)
            new_game = Clobber(new_board, self.game.player_color)
            new_game.switch_player()
            move_value = self.minimax(self.depth - 1, alpha, beta, False)
            if move_value > best_value:
                best_value = move_value
                best_move = move
        return best_move
