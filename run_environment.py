import os
import json
import copy
import sys
sys.path.insert(0, 'ai_optimizer')
import numpy as np
from production_plant_environment.env.production_plant_environment_v0 import ProductionPlantEnvironment
from utils.learning_policies_utils import initialize_agents
from utils.graphs_utils import RewardVisualizer



#CONFIG_PATH = "ai_optimizer/configs/simulator_config_3units.json"
CONFIG_PATH = "ai_optimizer/configs/simulator_config_siemens_20units_SAC.json"
SEMI_MDP_CONFIG_PATH = "config/semiMDP_reward_config.json"
LEARNING_CONFIG_PATH = "ai_optimizer/configs/learning_config.json"
MQTT_HOST_URL = 'localhost'

with open(CONFIG_PATH) as config_file:
    config = json.load(config_file)
with open(SEMI_MDP_CONFIG_PATH) as config_file:
    semiMDP_reward_config = json.load(config_file)
with open(LEARNING_CONFIG_PATH) as config_file:
    learning_config = json.load(config_file)

test_model = config['test_model']
test_model_name = config['test_model_name']
test_model_n_episodes = config['test_model_n_episodes']
n_agents = config['n_agents']
n_products = config['n_products']
n_episodes = config['n_episodes']
num_max_steps = config['num_max_steps']
n_production_skills = config['n_production_skills']
nothing_action = n_production_skills + 5
algorithm = config['algorithm']
export_trajectories = config['export_trajectories']
output_log = config['output_log']
custom_reward = config['custom_reward']
supply_action = config['supply_action']
discount_factor = config['gamma']
baseline_path = config['baseline_path']
one_hot_state = config['one_hot_state']
agent_connections = config["agents_connections"]
loop_threshold = learning_config["loop_threshold"]
checkpoint_frequency = learning_config["checkpoint_frequency"]
shaping_value = learning_config["shaping_value"]
use_masking = learning_config["action_masking"]
constant_positive_shaping_flag = learning_config["constant_positive_shaping_flag"]
positive_shaping_constant = learning_config["positive_shaping_constant"]
positive_shaping_coefficient = learning_config['positive_shaping_coefficient']
time_shaping_coefficient = learning_config['time_shaping_coefficient']


def get_rllib_state(state, old_state, one_hot_state=False, use_masking=False):
    # next_skill , previous_agent, threshold_detected
    obs_rllib = []
    next_skill = state["products_state"][0,:, 0].tolist().index(1)
    previous_agent = old_state["current_agent"]
    if one_hot_state:
        next_skill_ohe = np.zeros(n_production_skills)
        next_skill_ohe[next_skill] = 1
        next_skill = next_skill_ohe
        previous_agent_ohe = np.zeros(n_agents)
        previous_agent_ohe[previous_agent] = 1
        previous_agent = previous_agent_ohe
    obs_rllib.extend(next_skill)
    obs_rllib.extend(previous_agent)
    obs_rllib = np.array(obs_rllib).flatten().tolist()
    if use_masking:
        action_mask = get_action_mask(state)
        obs_rllib = {
            "observations": obs_rllib,
            "action_mask": action_mask.tolist()
        }
    return obs_rllib

def get_action_mask(state):
    # Only works for 3 agents now
    current_agent = state["current_agent"]
    if current_agent == 1:
        action_mask = np.ones(2)
        if np.random.rand() < 0.5:
            action_mask[np.random.choice(2)] = 0
    else:
        action_mask = np.ones(1)
    return action_mask

def extract_kpi(skill): # TODO remove since it is done inside the env
    fact_duration = 0
    fact_energy = 0
    kpi_duration = skill["Duration"]
    idle_energy_consumption = skill['IdleEnergyConsumption']
    dynamic_energy_consumption = sum([behavior['DynamicEnergyConsumption']
                                      for behavior in skill['Behaviors']])
    kpi_energy = idle_energy_consumption + dynamic_energy_consumption
    kpi = fact_duration * kpi_duration + fact_energy * kpi_energy
    return kpi


def shape_reward(overall_kpi, production_kpi, positive_shaping_coefficient, time_shaping_coefficient):

    """
    Shape the reward
    - overall_ : referring to the entire history of the product. (e.g. "Duration"/"EnergyConsumption" since start of production)
    - production_ : referring only to the self.production_skills part of the product's history (e.g. "Duration"/"EnergyConsumption" spent executing production skills)
    - _kpi: actual value is a custom combination of the two KPI types "Duration" and "EnergyConsumption". Check combination used in self.extract_kpi().
    """
    return positive_shaping_coefficient * production_kpi - time_shaping_coefficient * overall_kpi


def compute_reward_rllib(reward, skill, non_production_skill):
    """ Compute the reward in a Semi-MDP fashion for RLlib"""
    overall_kpi = reward
    if skill["Skill"] not in non_production_skill:
        production_kpi = reward
    else:
        production_kpi = 0
    return overall_kpi, production_kpi


def convert_action(action, agent):
    return action  # action is converted to port directly in agent


def get_cppu_name(agent):
    return f'cppu_{agent}'

OUTPUT_PATH = config['output_path']
TRAJECTORY_PATH = config['trajectory_path']
communicator = initialize_agents(n_agents, algorithm, n_episodes, custom_reward,
                          config['available_actions'], config['agents_connections'], mqtt_host_url=MQTT_HOST_URL)

model_path = "models/rllib/"
if not os.path.exists(model_path):
    os.makedirs(model_path)
run = 1
env = ProductionPlantEnvironment(config, run=run, model_path=model_path, semiMDP_reward_config=semiMDP_reward_config)
reward_visualizer = RewardVisualizer(n_agents)
performance = {}

cppu_names = [f'cppu_{i}' for i in range(n_agents)]
non_production_skills = []
if test_model:
    file_log_name = f"{model_path}/test_logs.txt"
else:
    file_log_name = f"{model_path}/training_logs.txt"
with open(file_log_name, 'w') as file:
    file.write(f"")

if test_model:
    n_episodes = test_model_n_episodes

for episode in range(n_episodes):
    if output_log:
        # create log
        with open(f"{OUTPUT_PATH}_{episode}.txt", 'w') as file:
            file.write('')

    if export_trajectories:
        # create trajectory
        trajectories = {f'Episode {episode}, Product {product}': [] for product in range(n_products)}
        with open(f"{TRAJECTORY_PATH}_{episode}.json", 'w') as outfile:
            json.dump(trajectories, outfile, indent=6)

    with open(file_log_name, 'a') as file:
        file.write(f"Starting episode {episode+1}\n")

    # used for old reward counting (before new graph)
    agents_rewards = [[] for _ in range(n_agents)]
    # used for reward computation for new graph
    agents_rewards_for_plot = {}
    for i in range(n_agents):
        agents_rewards_for_plot[i] = 0
    if custom_reward == 'reward5':
        trajectory_for_semi_MDP = []
    previous_rewards = {}
    state = env.reset()
    old_state = copy.deepcopy(state)
    episode_id = f"E{episode}"
    communicator.publish_episode_management('Start', episode_id)
    communicator.sync_episode()

    for step in range(num_max_steps):
        action_selected_by_algorithm = False

        if np.all(np.array(state['action_mask'][state['current_agent']]) == 0) \
                or state['agents_busy'][state['current_agent']][0] == 1:
            # if no actions available -> do nothing
            action = nothing_action
        else:
            action_selected_by_algorithm = True
            cppu_name = get_cppu_name(state["current_agent"])
            obs = get_rllib_state(state, one_hot_state=one_hot_state, old_state=old_state, use_masking=use_masking)
            #obs_rllib = []
            if cppu_name in previous_rewards:
                overall_kpi, production_kpi = previous_rewards[cppu_name]
                reshaped_reward = shape_reward(overall_kpi, production_kpi, positive_shaping_coefficient, time_shaping_coefficient)
            else:
                reshaped_reward = None  # first time
            previous_rewards[cppu_name] = (0, 0) # reset accumulators
            threshold_detected = state["time"] > loop_threshold  # TODO: implement a  real threshold check
            communicator.set_action_barrier(cppu_name)
            communicator.send_state_and_previous_reward(cppu_name, obs, reshaped_reward, threshold_detected)
            # only works for single product as we need to wait for the action on this agent
            raw_action = communicator.receive_action(cppu_name)
            action = convert_action(raw_action, state["current_agent"])
        # Here send the message to the workers
        previous_state = state
        state, reward, done, info, product = env.step(action)
        if action_selected_by_algorithm:
            for agent in cppu_names:
                if agent in previous_rewards:
                        overall_kpi = reward
                        production_kpi = 0
                        if info["production_skill_executed"]:
                            production_kpi = info["production_skill_duration"]
                            communicator.publish_skill_execution(product, info['skill']) 
                        previous_rewards[agent] = (previous_rewards[agent][0] + overall_kpi,
                                                   previous_rewards[agent][1] + production_kpi)
        if done:
            break
    if checkpoint_frequency is None:
        save_checkpoint = False
    else:
        save_checkpoint = True if (episode + 1) % checkpoint_frequency == 0 or episode == n_episodes - 1 else False
    last_rewards = {}
    for agent in cppu_names:
        if agent in previous_rewards:
            overall_kpi, production_kpi = previous_rewards[agent]
            reshaped_reward = - shape_reward(overall_kpi=overall_kpi, production_kpi=production_kpi,
                                             shaping_value=shaping_value)
        else:
            reshaped_reward = None
        last_rewards[agent] = reshaped_reward
    communicator.publish_episode_management('End', episode_id, produced_product=None, save_checkpoint=save_checkpoint,
                                            last_rewards=last_rewards)
    communicator.sync_episode()
    print(f'Finished Episode {episode}!!')
    # semi MDP reward postponed computation (at the end of the episode)
    #
    # performance[episode] = {}
    # performance[episode]['episode_duration'] = state['time']
    # performance[episode]['agents_reward_for_plot'] = agents_rewards_for_plot