from typing import TypedDict


class TrainingCheckpoint(TypedDict):
    neural_net_name: str
    neural_net_state_dict: dict
    optimizer_state_dict: dict
