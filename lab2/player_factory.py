from typing import Callable, Dict


class PlayerFactory:
    _creators: Dict[str, Callable] = {}

    @classmethod
    def register(cls, key: str, creator: Callable):
        cls._creators[key.lower()] = creator

    @classmethod
    def create(cls, player_type: str, color: str, **kwargs):
        creator = cls._creators.get(player_type.lower())
        if not creator:
            raise ValueError(f"Unknown player type '{player_type}'")
        return creator(color=color, **kwargs)
