from typing import List, Optional, Type, Union
import numpy as np
import math
import os
import json
import glob

INF_VALUE = -9999999999



def get_string_observation(observation):
    """This method receives the observation in a tuple format and convert it to a string """
    if (not isinstance(observation, tuple) and
        not isinstance(observation, list)):
        return observation
    obs_string = ''
    for i in range(len(observation)):
        obs_string += str(observation[i])
        if i != len(observation) - 1:
            obs_string += ", "
    return obs_string


class Custom_value_based_policy():

    def __init__(self, cppu_name, agent, observation_space=None,
                 action_space=None, config=None, logger=None,
                 policy_seed=None):

        self.cppu_name = cppu_name
        self.agent = agent
        self.config = config
        self.observation_space = None

        self.action_space = action_space
        self.logger = logger

        self.values = {}
        self.q_init = self.config["q_init"]

        self.history_transitions = []
        self.last_transition = {}

        self.set_exploration()

        self.episode = 0

        np.random.seed(policy_seed)
        logger.info(f"Initialized policy with policy_seed={policy_seed}")

    def reset(self):
        """Reset the policy history"""
        self.history_transitions = []
        self.last_transition = {}

    def set_exploration(self):
        self.last_eid = 'E0'
        if self.config["exploration"] == 'eps-greedy':
            self.exploration_param = self.config["epsilon"] 
            self.epsilon_for_inference = self.config["epsilon_for_inference"]
        elif self.config["exploration"] == 'softmax':
            self.exploration_param = self.config["tau"]
        self.delta_exploration_param = self.exploration_param * \
            self.config["exploration_var_rate"] / self.config["episodes"]

    def update_exploration_param(self):
        self.exploration_param -= self.delta_exploration_param
    
    def modify_values(self):
        self.logger.info('Updating Unseen Values')
        for key, values in self.values.items():
            mod_values = np.array(values['Q'])
            idx_mod = mod_values == self.q_init
            mod_values[idx_mod] = INF_VALUE
            self.values[key]['Q'] = list(mod_values)            
        
    def set_final_values(self, observation, final_value = 0.):
        key = get_string_observation(observation)
        if key not in self.values.keys():
            self.values[key] = {}
        self.values[key]['Q'] = [final_value] * self.action_space
        self.logger.info(f'Q({key}, :) set to {final_value}')

    def get_action(self, eid, obs, action_set = None, inference_mode = False):
        self.logger.info('Episode {}. Called get_action with obs {}'.format(eid, obs))
        if eid != self.last_eid:
            self.update_exploration_param()
            self.last_eid = eid
        
        if self.config["exploration"] == 'eps-greedy':
            epsilon = self.exploration_param if not inference_mode else self.epsilon_for_inference
            observation_not_found = get_string_observation(obs) not in self.values.keys()
            if np.random.rand() < epsilon or observation_not_found:
                action = np.random.randint(low=0, high=self.action_space) if action_set is None else int(np.random.choice(action_set))
                self.logger.info(f'Acting Randomly (Observation Not Found): {observation_not_found}')
            else:
                action_values = self.values[get_string_observation(obs)]['Q']
                action = self.max_break_ties(action_values)
                self.logger.info(f'Acting Greedily over {action_values}')
        elif self.config["exploration"] == 'softmax':
            if get_string_observation(obs) not in self.values.keys():
                action = np.random.randint(low=0, high=self.action_space) if action_set is None else int(np.random.choice(action_set))
            else:
                values = np.array(self.values[get_string_observation(obs)]['Q'])
                a = np.exp(values / self.exploration_param)
                prob = np.nan_to_num(a / np.sum(a))
                action = np.random.choice(self.action_space, p=prob)
        return action
    
    def max_break_ties(self, values):
        values_array = np.array(values)
        array = np.flatnonzero(values_array == values_array.max())
        return array[0]
                
                
    def learn_on_batch(self, samples, agents_information):
        rewards = samples['rewards']
        previous_observation = samples['previous_observation']
        actions = samples['action']

        self.last_transition = {"obs": previous_observation,
                                "act": actions,
                                "reward": rewards}
        self.history_transitions.append(self.last_transition)

        self.update_values(observation=previous_observation,
                           action=actions,
                           reward=rewards,
                           agents_information=agents_information)

    def update_values(self, observation, action, reward, agents_information):
        pass

    def get_values(self, cppu, observation):
        """Read fromNext cppu (cppu) the values"""
        values = self.agent.get_whiteboard(cppu)['Q_values']
        self.logger.info(f'{cppu}: Q-table received is -> {values}')
        if get_string_observation(observation) not in values.keys():
            self.logger.info(f'Q-table of {cppu} does not contain the key {observation}')
            return self.q_init
        next_values = values[get_string_observation(observation)]['Q']
        self.logger.info(f'Retrieved Q-values of {cppu} --> Q({observation}) = {next_values}')
        max_next_q_value = np.max(next_values)
        self.logger.info(f'Q-Max is {max_next_q_value}')
        return max_next_q_value

    def log_returns(self, episode_id=None, reward=0):
        pass

    def compute_learning_rate(self, observation, action):
        if self.update_learning_rate:
            T = self.values[get_string_observation(observation)]['T'][action]
            new_lr = self.lr / (T ** self.lr_decreasing_factor)
            self.logger.info(f'Learning rate update for o = {observation}, ' +
                              f'a = {action} with T={T} --> new_lr = {new_lr}')
            return new_lr
        else:
            return self.lr

    def get_next_cppu_name(self, next_port):
        cppu_config = self.agent.config['plant_layout'][self.cppu_name]
        for connection in cppu_config['connections']:
               if connection['LocalPort'] == next_port:
                    cppu_name_next = connection['RemoteEquipment']
        return cppu_name_next
        
    def extract_information_from_neighbours(self, observation, port):
        pass

    def end_episode(self, episode_id, product_info, save_checkpoint, out_dir):

        if product_info is not None:
            last_skill = product_info['States']['SkillHistory'][-1]['Skill']
            last_cppu = product_info['States']['SkillHistory'][-1]['Cppu']
            if last_skill == 'store' and last_cppu == self.agent.cppu_name:
                previous_observation = self.agent.get_previous_observation(product_info)
                self.logger.info(f'END EPISODE ' + f'Previous observation is {previous_observation}')
                self.set_final_values(previous_observation, final_value=0.)

        if save_checkpoint:
            self.logger.info('Saving checkpoint')
            self.save(out_dir, episode_id)
        
        self.episode += 1

    def save(self, out_dir, episode_id):

        ep_label = episode_id.split('E')[-1]
        trained = ''
        model_dir = os.path.join(out_dir, f'models')

        if int(ep_label) == self.config["episodes"]-1:
            trained = 'trained'

        os.makedirs(model_dir, exist_ok=True)
        ckp_list = glob.glob(os.path.join(
            model_dir,
            f"{self.cppu_name}_DistQ_*.pkl"))
        self.logger.debug(f'ckp_list is {ckp_list}')
        for ckp_path in ckp_list:
            os.remove(ckp_path)
        self.logger.info(f'Saved Model')
        with open(os.path.join(
            model_dir,
            f"{self.episode}_{self.cppu_name}_DistQ_{trained}.pkl"), 'w') as f:
            f.write(json.dumps(self.values))

    def load(self, out_dir):
        ckp_path = f"{out_dir}/models/{self.cppu_name}_DistQ*.pkl"
        model_list = glob.glob(ckp_path)
        self.logger.info(f'model_list is {model_list}')
        if len(model_list) > 0:
            model_path = model_list[0]
            try:
                with open(model_path, 'r') as f:
                    self.values = json.loads(f.read())
                self.logger.info(f'{model_path} successfully read!')

                self.episode = int(model_path.split('_')[1])
                self.logger.info(f'self.episode updated to {self.episode}')

            except Exception as e:
                self.logger.info(f'Unable to load {model_path}')
                self.logger.error(e)
                raise e
        else:
            self.logger.info(f"Unable to restore checkpoint. no match for {ckp_path}")



class Dist_Q_policy(Custom_value_based_policy):

    def __init__(self, cppu_name, agent, observation_space=None,
                 action_space=None, config=None, logger=None, policy_seed=None):

        super().__init__(cppu_name, agent,
                         observation_space=observation_space,
                         action_space=action_space,
                         config=config, logger=logger,
                         policy_seed=policy_seed)
        
        self.lr = self.config["lr"]
        self.gamma = self.config["gamma"]
        self.update_learning_rate = self.config["update_lr"]
        if self.update_learning_rate:
            self.lr_decreasing_factor = self.config["lr_decreasing_factor"]

    def update_values(self, observation, action, reward, agents_information):
        # agents_information are q_value_next_max ONLY
        # If this observation it's not in the q_value table
        self.logger.info('Called update_values with transition ' +
                         '(obs={},action={},reward={})'. \
                            format(observation, action, reward))
        if get_string_observation(observation) not in self.values.keys():
            # Init for that key a list of zeros with lenght equal to the number 
            # of ports (action space)
            self.values[get_string_observation(observation)] = {}
            self.values[get_string_observation(observation)]['Q'] = \
                [self.q_init] * self.action_space
            self.values[get_string_observation(observation)]['T'] = \
                [0] * self.action_space
            self.logger.info('No key inside Q table --> init: Q({}) = {}'. \
                format(observation,
                       self.values[get_string_observation(observation)]['Q']))
            
        # Save the current q value
        current_q_value = self.values \
            [get_string_observation(observation)]['Q'][action]
        self.values[get_string_observation(observation)]['T'][action] += 1
        self.logger.info('Current values: {} --> Q={}, T={}'. \
                format(observation,
                    self.values[get_string_observation(observation)]['Q'],
                    self.values[get_string_observation(observation)]['T']))

        # Get the learning rate to use
        lr = self.compute_learning_rate(observation, action)
        # Update the value for the current observation and action pair
        next_q_value = (1 - lr) * current_q_value + lr * \
            (reward + self.gamma * agents_information)
        self.logger.info('next_q_value = (1 - {}) * {} + {} * ({} + {} * {}) = {}'. \
                         format(lr, current_q_value, lr, reward, self.gamma,
                                agents_information, next_q_value))
        # Update
        self.values[get_string_observation(observation)]['Q'][action] = np.round(next_q_value, 3)
        self.logger.info('New values: {} --> Q={}, T={}'.format(observation,self.values[get_string_observation(observation)]['Q'], self.values[get_string_observation(observation)]['T']))

    def extract_information_from_neighbours(self, observation, port):
        cppu_name_next = self.get_next_cppu_name(next_port=port)
        self.logger.info(f'Next cppu is {cppu_name_next} at port {port}')
        q_value_next_max = self.get_values(cppu_name_next, observation)
        return q_value_next_max