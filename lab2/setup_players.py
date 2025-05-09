from heuristics import HeuristicFactory, HeuristicType
from player_factory import PlayerFactory
from players import HumanPlayer, WebSocketHumanPlayer, AIPlayer


def register_players():
    """
    Register player types with the PlayerFactory.
    """
    # Register human, websocket human, and AI players

    PlayerFactory.register('human', lambda color, **kwargs: HumanPlayer(color))
    PlayerFactory.register('ws_human', lambda color, websocket, **kwargs: WebSocketHumanPlayer(color, websocket))
    PlayerFactory.register('ai', lambda color, depth=3, heuristic_type=HeuristicType.MOBILITY, **kwargs:
        AIPlayer(color, depth, HeuristicFactory.create(heuristic_type)))
