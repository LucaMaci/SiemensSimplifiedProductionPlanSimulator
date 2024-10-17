from gymnasium import spaces
from utils.trajectories_management import flatten_dict_values
import numpy as np

# define MDP used by FQI agents
class MDP():
    def __init__(self, state_dim, actions):
        self.action_space = spaces.Discrete(len(actions))
        self.state_dim = state_dim
        self.action_dim = 1
        self.gamma = 0.99

def get_FQI_state(state, observable_neighbours, n_products):
    state['agents_state'] = [state['agents_state'][i] for i in range(len(state['agents_state'])) if i in observable_neighbours]
    products_mask = np.zeros(n_products)
    for agent_state in state['agents_state']:
        if max(agent_state) == 1:
            products_mask[np.argmax(agent_state)] = 1
    for i in range(len(state['products_state'])):
        if products_mask[i] == 0:
            state['products_state'][i] = np.zeros_like(state['products_state'][i]).tolist()
    return flatten_dict_values(state)