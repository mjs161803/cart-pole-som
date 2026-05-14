from utils.gng import GNG

class TrinityAgent():
    def __init__(self, observation_dim, action_dim):
        self._previous_action = None
        self._current_observation = None
        self._current_instruction = None

        self._observation_map = GNG(input_dim=observation_dim, max_nodes=100, lambda_=10, alpha=0.5, global_decay=0.99)
        self._goal_map = GNG(input_dim=action_dim, max_nodes=100, lambda_=10, alpha=0.5, global_decay=0.99)