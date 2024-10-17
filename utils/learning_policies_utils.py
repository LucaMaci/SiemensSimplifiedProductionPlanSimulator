from learning_policies.learning_policies import DistributedQLearningAgent, LPIAgent, FQIAgent
import numpy as np
import json
from typing import Literal
from ai_optimizer.communicator import Communicator
DIST_Q_CONFIG_PATH = "config/DistQ_config.json"
LPI_CONFIG_PATH = "config/LPI_config.json"
FQI_CONFIG_PATH = "config/FQI_config.json"

# given number of agents and algorithm return a list of istances of agents of that specific algorithm
def initialize_agents(n_agents, algorithm, n_episodes, reward_type, available_actions, agents_connections,
                      mqtt_host_url=None):
    model_units_folder = f'{n_agents}_units'
    if algorithm == "DistQ":
        with open(DIST_Q_CONFIG_PATH) as config_file:
            config = json.load(config_file)
        return [DistributedQLearningAgent(config, model_units_folder, i, n_episodes, reward_type, available_actions,
                                          agents_connections) for i in range(n_agents)], config['update_values'],\
            config['policy_improvement']
    elif algorithm == "LPI":
        with open(LPI_CONFIG_PATH) as config_file:
            config = json.load(config_file)
        return [LPIAgent(config, model_units_folder, i, n_episodes, reward_type, available_actions,
                         agents_connections) for i in range(n_agents)], config['update_values'],\
            config['policy_improvement']
    elif algorithm == "FQI":
        with open(FQI_CONFIG_PATH) as config_file:
            config = json.load(config_file)
    elif algorithm == "rllib":
        communicator = Communicator(n_agents, mqtt_hostname=mqtt_host_url)

        return communicator
    
# TO-DO: check if move these following 2 functions in DistributedQLearningAgent class

# given a state return an observation which consists of the product that the current agent has
# and of the current skill progress of that specific product (FOR ONLINE TRAINING ONLY)
def get_agent_state_and_product_skill_observation_DISTQ_online(agent, state):
    curr_agent_state = state['agents_state'][agent]
    curr_product_skills = state['products_state'][np.argmax(curr_agent_state)]
    return f'{curr_agent_state}, {curr_product_skills}'

# given a state return an observation which consists of the product that the current agent has
# and of the current skill progress of that specific product (FOR OFFLINE TRAINING ONLY)
def get_agent_state_and_product_skill_observation_DISTQ_offline(state):
    return f'{state["curr_agent_state"]}, {state["curr_product_skills"]}'

# def send_action(exchange_dict, action, skill: Literal['transport', 'defer', 'buffer']='transport'):
#     exchange_dict['port'] = action
#     exchange_dict['skill'] = skill
#     mqtt_topic = '/'.join(['PathSelection',
#                            exchange_dict['cppu'],
#                            exchange_dict['product'],
#                            exchange_dict['identifier']])
#     with self.mqtt_pub_client_lock:
#         self.mqtt_pub_client.publish(mqtt_topic,
#                                      json.dumps({"port": {"id": action},
#                                                  "skill": skill}))
#     return f'Product [{exchange_dict["product"]}] Port {action}'





