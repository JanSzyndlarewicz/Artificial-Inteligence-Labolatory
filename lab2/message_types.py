

class MessageType(str):
    REGISTER = "register"
    REQUEST_MOVE = "request_move"
    MAKE_MOVE = "make_move"
    MOVE_MADE = "move_made"
    GAME_STARTED = "game_started"
    GAME_OVER = "game_over"
    CREATE_GAME = "create_game"
    GAME_CREATED = "game_created"
    OPPONENT_DISCONNECTED = "opponent_disconnected"
    WAITING = "waiting_for_opponent"
