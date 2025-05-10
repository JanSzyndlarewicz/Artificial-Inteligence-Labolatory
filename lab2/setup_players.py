from heuristics import HeuristicFactory, HeuristicType
from player_factory import PlayerFactory
from players import HumanPlayer, WebSocketPlayer, AIPlayer


def register_players():


    PlayerFactory.register('human', lambda color, **kwargs: HumanPlayer(color))
    PlayerFactory.register('websocket', lambda color, websocket, **kwargs: WebSocketPlayer(color, websocket))
    PlayerFactory.register('ai', lambda color, depth=3, heuristic_type=HeuristicType.MOBILITY, **kwargs:
        AIPlayer(color, depth, HeuristicFactory.create(heuristic_type)))
