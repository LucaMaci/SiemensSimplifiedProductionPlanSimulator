import os
import gymnasium as gym
import time
import threading
import argparse
import numpy as np
import random
import json
from typing import Literal
import paho.mqtt.client as mqtt
from agent_simulator import AgentSImulator
from ray.rllib.env.policy_client import PolicyClient
from custom_policies import Dist_Q_policy
from logging_formatter import setup_logger


SERVER_BASE_TOPIC = 'AI'
SERVER_READY_TOPIC = 'ServerReady'
SEED = 100000


def make_pretty_log_entry(log_entry):
    padding = '*' * len(log_entry)
    pretty_log_entry = '\n'.join(['', padding, log_entry, padding])
    return pretty_log_entry


class Agent(AgentSImulator):
    """
    Agent class containing the agent deployment for
    trainig and inference on RL systems.
    """

    def __init__(
        self, 
        cppu_name: str = None,
        learning_config: str = None,
        simulator_config: str = None, 
        product_config: str = None,
        serverport: str = None, 
        seed: int = None,
        mqtt_host_url: str = None, 
        gateway_url: str = None, 
        out_dir: str = None
    ) -> None:

        self.logger = setup_logger(
            f'Agent::{cppu_name}::PID{os.getpid()}'
        )

        super().__init__(
                cppu_name, 
                learning_config=learning_config,
                config=simulator_config,
                products=product_config,
                logger=self.logger,
                mqtt_host_url=mqtt_host_url,
                gateway_url=gateway_url)
        
        self.cppu_name = cppu_name
        self.learning_config = learning_config
        self.client_server = self.learning_config['client_server']
        self.algorithm_class = self.learning_config['algorithm_class']
        self.off_policy = self.learning_config['off_policy']
        self.loop_threshold = self.learning_config['loop_threshold']
        self.baseline = self.learning_config['baseline']
        self.action_masking = self.learning_config['action_masking']
        self.action_masking_number = self.learning_config['action_masking_number']
        self.out_dir = out_dir

        # Setting up Clients for MQTT communication
        self.mqtt_pub_client = mqtt.Client()
        self.mqtt_pub_client_lock = threading.Lock()
        self.mqtt_sub_client = mqtt.Client()

        # Setting up barriers for MQTT sync
        self.reward_barriers = {'variant': {}, 'path': {}}
        self.variant_history_dict_lock = threading.Lock()
        self.path_history_dict_lock = threading.Lock()
        self.server_ready = threading.Barrier(2)
        self.client_ready = threading.Barrier(2)

        # Setting up additional structures for exhaning info from classes
        self.variant_history_dict = {}
        self.path_history_dict = {}
        self.exchange_dicts = {'variant': {}, 'path': {}}
        self.trajectory = {}
        self.eval_trajectory = {}
        self.transported_products = set()
        self.skill_histories = {}

        # Setting up RL-specific attributes
        self.action_dimensions = len(self.ports)
        self.action_space = gym.spaces.Discrete(self.action_dimensions)
        self.logger.info('Ports: {}'.format(self.ports))
        self.logger.info('Ports Available: {}'.format(self.action_dimensions))
        self.last_observation = None
        self.episode_identifier = None
        self.dummy_reward = 0.0

        # Managing seeds for reproducibility
        np.random.seed(int(seed))
        self.policy_seed = np.random.randint(SEED)
        

        # Setting Up MQTT Clients
        self.setup_mqtt_clients()
        # Setting Up clients containing algorithms
        if self.client_server:
            self.client = self.connect_rllib_client(
                observation_space=self.get_observation_space(),
                action_space=self.get_action_space(),
                server_address=self.learning_config["SERVER_ADDRESS"],
                connection_port=serverport,
                inference_mode_rllib=self.learning_config["inference_mode"],
                update_interval=self.learning_config["update_interval"],
                behavioral_epsilon=learning_config["behavioral_epsilon"]
            )
        # Learning with custom algorithm (Dist-Q only)
        elif self.algorithm_class == 'Dist_Q':
            self.client = Dist_Q_policy(
                self.cppu_name, self,
                observation_space=None,
                action_space=self.action_dimensions,
                config=self.learning_config,
                logger=self.logger,
                policy_seed=self.policy_seed
            )
        else:
            self.logger.error(f'Algorithm Class {self.algorithm_class} not implemented')
            raise NotImplementedError
        self.client_ready.wait()
        self.update_whiteboard()
        self.logger.info('Agent Launched')

    def wait_for_server(self):
        self.logger.info("Waiting for Server")
        self.server_ready.wait()
        self.logger.info("Server Ready")

    def connect_rllib_client(self, 
                             observation_space=None,
                             action_space=None,
                             server_address=None,
                             connection_port=None,
                             inference_mode_rllib=None,
                             update_interval=None,
                             behavioral_epsilon=None):
        # Before launching clients, servers should be ON
        self.wait_for_server()
        # Build RLLib spaces & update interval (for management during training)
        self.observation_space = observation_space
        self.logger.info('Observation Space RLlib: {}'.format(self.observation_space))
        self.action_space = action_space
        self.logger.info('Action Space RLlib: {}'.format(self.action_space))
        self.update_interval = update_interval
        self.behavioral_epsilon = behavioral_epsilon
        try: 
            self.logger.info('Trying a connection on {} @ {}'.format(server_address, connection_port))
            client = PolicyClient(
                server_address + str(connection_port),
                inference_mode=inference_mode_rllib,
                update_interval=update_interval
            )
            self.logger.info('Client connected @ {}'.format(connection_port))
        except Exception as e:
            self.logger.error('Client connection failed @ {}: {}'.format(connection_port,e))
        return client


    def learning_step(self, observation, reward):
        """
        Performs Learning step once handle_path_selection message is received
        """
        # No reward barrier needed now?
        # self.reward_barriers['path'] = threading.Barrier(2)
        # self.logger.debug(f'Reward Barriers Created')
        eid = self.episode_identifier
        self.logger.info('Learning_step: Requesting action and learning')
        if self.client_server: 
            # extract reward of previous interaction and store them for learning at ending episode
            self.update_last_reward(reward)
            self.last_observation = observation
        elif self.algorithm_class == 'Dist_Q':
            raise ValueError("Dist Q not supported")
            #self.last_observation = observation[0]

        action = self.infer_action(observation)
        port = self.get_port(int(action))
        self.publish_port(port)
        self.logger.info('Action: {}'.format(action))

        # Wait until the reward can be collected
        # No reward needed here for RLlib
        # self.logger.debug(f' REWARD[{reward}] ' + 'Barriers locked ...')
        # self.reward_barriers['path'].wait()
        # reward = self.exchange_dicts['path']['reward']
        # self.logger.debug(f' REWARD[{reward}] ' + 'Barriers Unlocked ...')
        # self.reward_barriers['path'].pop(mqtt_id)
        # self.logger.debug(f'REWARD[{reward}] ' + 'Barriers Deleted ...')
        
        agents_information = None
        if not self.client_server and self.algorithm_class == 'Dist_Q':
            raise ValueError("Dist_Q not implemented here")
            # extract info from other agents for learning by using the observation AFTER skill execution
            # next_agent_obs = self.get_next_agent_observation(observation[1], port)
            # self.logger.info('Next Agent Observation is {}'.format(next_agent_obs))
            # agents_information = self.client.extract_information_from_neighbours(next_agent_obs, port)
            # samples = {'rewards': reward, 'action':action, 'previous_observation': observation[0], 'info': None}
            # # learn on batch for non-rllib agents
            # self.client.learn_on_batch(samples, agents_information)

        self.trajectory[str(len(self.trajectory.keys()))] = {
            'action': action, 
            'observation': observation,
            'rewards': reward,
            'info': {
                'port': port,
                'eid': eid,
                # 'product_id': product_info,
                'agent_information': agents_information
            }
        }

        # self.update_whiteboard()
        return int(action)

    def infer_action(self, obs, inference_mode=False):
        """
        From Observations return the action after publishing the port
        Action inference in inference_mode, get actions according to final policy 
        Input: 
            observation: observation custom/RLlib based
            identifier: id of the episode for rllib
        Output: int(action)
        """
        if self.client_server:
            # Observation is RLLIb
            observation = obs
            if not self.off_policy or inference_mode:
                action = self.client.get_action(self.episode_identifier, observation)
            else:
                raise ValueError("Baseline infer not supported")
                # behav = 'epsilon_min_hop' if self.baseline == 'min_hop' else 'random'
                # action = self.infer_action_with_baseline(product_info, behavior=behav)
                # self.client.log_action(self.episode_identifier, observation, action)
        else:
            raise ValueError("Only Client-Server supported")
            # # Observation is observation_before
            # observation = obs[0]
            # if self.action_masking:
            #     min_masking = min(self.action_masking_number, self.action_dimensions)
            #     observation_baseline = self.get_observation_baseline(product_info)
            #     masked_actions = np.argsort(observation_baseline)[:min_masking]
            #     action = self.client.get_action(self.episode_identifier, observation, action_set = masked_actions, inference_mode = inference_mode)
            # else:
            #     action = self.client.get_action(self.episode_identifier, observation, inference_mode = inference_mode)
        self.logger.info(f'The action returned is {action}')
        return action

    def reset(self):
        '''Resetting of Agent in case of restart'''
        self.last_observation = None
        self.counter_dict = {}
        if not self.client_server:
            self.client.reset()
        self.update_whiteboard()

    def update_whiteboard(self):
        ''' 
        Update white-board values for sharing info between agents 
        '''
        if self.client_server:
            whiteboard = {'counter_dict': self.counter_dict}
        elif self.algorithm_class == 'Dist_Q':
            whiteboard = {'Q_values' : self.client.values, 'counter_dict': self.counter_dict}
        self.set_whiteboard(self.cppu_name, whiteboard)

    def update_last_reward(self, reward):
        '''
        In case of Rllib before logging one action the reward for the previous one should be computed.
        If on-policy, log the reward of the previous action (if any)
        '''
        assert (len(self.trajectory) > 0 and reward is not None) or (len(self.trajectory) == 0 and reward is None)
        if reward is not None:
            last_sample_key = list(self.trajectory.keys())[-1]
            last_sample = self.trajectory[last_sample_key]
            self.client.log_returns(episode_id=last_sample['info']['eid'], reward=reward)

        # if len(self.trajectory) > 0:
            # Log the reward corresponding to the last action
            # last_sample_key = list(self.trajectory.keys())[-1]
            # last_sample = self.trajectory[last_sample_key]
            # Check if the last product id is again in the cppu
            # TODO modify to handle more products
            # with a single product this check is not needed
            # if True: #last_sample['info']['product_id'] in list(self.get_products(self.cppu_name)):
                # Get the product info
                # product_info = self.get_product(self.cppu_name, last_sample['info']['product_id'][0])
                # last_reward = self.compute_reward_rllib(product_info['States']['SkillHistory'])
                # self.client.log_returns(episode_id=last_sample['info']['eid'], reward=last_reward)

    def handle_variant_selection(self, cppu, skill, mqtt_id):
        """
        Given the cppu, the skill and the identifier, this method select the variant of the skill
        """
        self.exchange_dicts['variant'][mqtt_id] = {}
        self.logger.debug(f'MQTT[{mqtt_id}] ' +
                         'VARIANT Exchange Dict Created ...')
        self.exchange_dicts['variant'][mqtt_id]['cppu'] = cppu
        self.exchange_dicts['variant'][mqtt_id]['skill'] = skill
        self.exchange_dicts['variant'][mqtt_id]['identifier'] = mqtt_id

        variant = 0
        self.publish_variant(cppu, variant, mqtt_id)
        self.publish_variant_learning_status(cppu, skill, mqtt_id)

    def handle_path_selection(self, state, reward, threshold_detected=False):
        """ 
        Given the cppu, this method asks for the port where to transport the product
        """

        self.exchange_dicts['path'] = {}

        # self.logger.debug(f'MQTT[{mqtt_id}] PATH Exchange Dict Created ...')
        # self.exchange_dicts['path']['cppu'] = cppu
        # self.exchange_dicts['path']['product'] = product

        # product_info = self.get_product(self.cppu_name, product)
        # self.transported_products.add(product)
        
        if not threshold_detected:
            if not self.baseline_mode:
                observation = self.prepare_observation(state)
                #self.return_observation_tuples(product_info, self.client_server, self.algorithm_class)
                if self.train_mode:
                    # Port publishing meanwhile managing information
                    self.learning_step(observation, reward)
                elif self.inference_mode:
                    raise ValueError("Only train_mode implemented") # TODO do we need this?
                    # action = self.infer_action(observation, inference_mode=self.inference_mode)
                    # port = self.get_port(int(action))
                    # self.publish_port(port)
                    # self.eval_trajectory[str(len(self.eval_trajectory.keys()))] = {
                    #     'action': action,
                    #     'observation': observation[0]}
                    # self.logger.debug(f"Updated eval_trajectory: {self.eval_trajectory}")
            else:
                raise ValueError("baseline mode not supported")
                # action = self.infer_action_with_baseline(product_info, behavior = self.baseline)
                # port = self.get_port(int(action))
                # self.publish_port(port, mqtt_id)
        else:
            fail_port = "-1"
            self.logger.info(f"Sending Port {fail_port} after Threshold Detection")
            # self.publish_port(fail_port, mqtt_id)

        with self.path_history_dict_lock:
            self.path_history_dict = self.exchange_dicts['path']

        # self.exchange_dicts['path'].pop(mqtt_id)
        self.logger.debug(f' PATH Exchange Dict Deleted ...')
        self.publish_path_learning_status(self.cppu_name)


    def infer_action_with_baseline(self, product_info, behavior = None):
        if behavior == 'random':
            action = np.random.randint(low=0, high=len(self.ports))
        elif behavior == 'min_hop':
            # Select the action that minimize the distance
            observation_baseline = self.get_observation_baseline(product_info)
            action = np.argmin(observation_baseline)
        elif behavior == 'epsilon_min_hop':
            # Select the action that minimize the distance with epsilon probability
            observation_baseline = self.get_observation_baseline(product_info)
            if np.random.rand() < self.behavioral_epsilon:
                action = np.random.randint(low=0, high=len(self.ports))
            else:
                action = np.argmin(observation_baseline)
            self.logger.info(f"Baseline Action {action}")
        return action


    def check_production_threshold(self, state):
        """ 
        Method to check whether the production has been going on too long since the last production skill 
        """
        if self.use_masking:
            state = state["observations"]
        return state[-1]
        # self.logger.info(f'Checking Production Threshold for product {product}')
        # skill_history = product_info['States']['SkillHistory']
        #
        # noprod_skills = ["transport", "buffer", "defer"]
        # count = 0
        # threshold_detected = False
        # if len(skill_history) > self.loop_threshold:
        #     for i in range(1, len(skill_history)):
        #         last_entry = skill_history[-i]
        #         if last_entry.get("Skill") in noprod_skills:
        #             count+=1
        #         else:
        #             break
        #
        # if count > self.loop_threshold:
        #     self.logger.info(f'Production Threshold Detected for product {product}')
        #     self.publish_production_threshold()
        #     threshold_detected = True
        # return threshold_detected

    def propagate_reward_to_variant_gym(self, variant_state, cppu, skill, identifier):
        """Propagate the reward to the variant environment"""
        # Check if all the parameters match up with the exchange dictionary
        # exchange_dict = self.exchange_dicts['variant'][identifier]
        self.logger.debug(f'MQTT[{identifier}] ' +
                         f'Variant State: {variant_state}')

    def propagate_reward_to_path_gym(self, transition):
        """Propagate the reward to the path environment"""
        self.logger.debug(f' ' +
                         f'Transition: {transition}')
        # product_info = self.get_product(cppu, product)
        # reward = self.compute_reward(product_info['States']['SkillHistory'])
        reward = transition["reward"]
        self.exchange_dicts['path']['reward'] = reward
        self.logger.debug(f'REWARD[{reward}] Barriers Locked ...')
        # self.reward_barriers['path'].wait()

    def setup_mqtt_clients(self):
        self.logger.debug(f'Setting up MQTT client ...')
        for mqtt_client in (self.mqtt_pub_client, self.mqtt_sub_client):
            mqtt_client.on_connect = self.on_connect
            mqtt_client.connect(self.get_mqtt_hostname())
        self.logger.debug(f'MQTT client set up ...')

        self.mqtt_sub_client.on_message = self.mqtt_sub_callback
        
        self.logger.debug(f'{self.cppu_name}: ' +
            'subscribing to the variant selection and variant state topic ...')
        # SIMULATOR-RELATED
        self.mqtt_sub_client.subscribe(f'VariantSelection/#')
        self.mqtt_sub_client.subscribe(f'VariantState/#')

        self.logger.debug(f'{self.cppu_name}: ' +
            'subscribing to the path selection and path state topic ...')
        self.mqtt_sub_client.subscribe(f'PathSelection/#')
        self.mqtt_sub_client.subscribe(f'PathState/#')
        self.logger.debug(f'{self.cppu_name}: ' +
            'subscribing to the training regime and episode management topic ...')
        
        # ORCHESTRATOR-RELATED
        self.mqtt_sub_client.subscribe(f'TrainingRegime/{self.cppu_name}')
        self.mqtt_sub_client.subscribe(f'EpisodeManagement/#')
        self.mqtt_sub_client.subscribe(f'AgentReadyRequest/#')
        self.mqtt_sub_client.subscribe(f'UpdateValues/#')
        mqtt_topic_server_ready = '/'.join([SERVER_BASE_TOPIC, SERVER_READY_TOPIC, f'{self.cppu_name}'])
        self.mqtt_sub_client.subscribe(mqtt_topic_server_ready)

        self.mqtt_sub_client.loop_start()
        self.mqtt_pub_client.loop_start()

    def on_connect(self, client, userdata, flags, rc):
        self.logger.debug(f'MQTT: connection status {rc}')

    def mqtt_sub_callback(self, client, userdata, message):
        parameters = message.topic.split('/')
        payload = message.payload.decode('utf-8')

        self.logger.debug(f'Received parameters: ' +
                          f'{parameters}, payload: {payload}')
        
        # if (parameters[0] == payload == 'VariantSelection'
        #     and parameters[1] == self.cppu_name):
        #     self.logger.debug(f'CPPU {self.cppu_name} ' +
        #                       'MQTT message VariantSelection ...')
        #     threading.Thread(target=self.handle_variant_selection,
        #                      args=parameters[1:]).start()
        # elif (parameters[0] == 'VariantState'
        #       and parameters[1] == self.cppu_name):
        #     self.logger.debug(f'CPPU {self.cppu_name} ' +
        #                       'MQTT message VariantState ...')
        #     threading.Thread(target=self.propagate_reward_to_variant_gym,
        #                      args=(payload, *parameters[1:])).start()
        if (parameters[0] == 'PathSelection'
              and parameters[1] == self.cppu_name):
            self.logger.debug(f'CPPU {self.cppu_name} ' +
                              'MQTT message PathSelection ...')
            payload = json.loads(payload)
            state = payload["state"]
            reward = payload["reward"]
            threshold_detected = bool(payload["threshold_detected"])
            threading.Thread(target=self.handle_path_selection,
                             args=[state, reward, threshold_detected]).start()
        elif (parameters[0] == 'PathState'
              and self.train_mode
              and parameters[1] == self.cppu_name):
            self.logger.debug(f'CPPU {self.cppu_name} ' +
                              'MQTT message PathState ...')
            threading.Thread(target=self.propagate_reward_to_path_gym,
                            args=payload).start()
        elif (parameters[0] == 'TrainingRegime'
              and parameters[1] == self.cppu_name):
            self.logger.info(f'{self.cppu_name}: received TrainingRegime message with payload {payload}')
            threading.Thread(target=self.update_training_regime,
                             args=[int(payload)]).start()
        elif (parameters[0] == 'EpisodeManagement'
              and parameters[1] == self.cppu_name
              and parameters[2] == 'Start'):
            payload = json.loads(payload)
            threading.Thread(target=self.start_episode,
                             args=[payload['episode_id']]).start() 
        elif (parameters[0] == 'EpisodeManagement'
              and parameters[1] == self.cppu_name
              and parameters[2] == 'End'):
            self.logger.debug(f'{self.cppc_name}: received end episode message!')
            payload = json.loads(payload)
            threading.Thread(target=self.end_episode,
                              args=[payload['episode_id'],
                                    payload['produced_product'],
                                    payload['save_checkpoint'],
                                    payload['last_reward']]).start()
        elif (parameters[0] == 'AgentReadyRequest'
              and parameters[1] == self.cppu_name
              and int(payload) == 1):
            self.logger.debug(f'{self.cppc_name}: received AgentReadyRequest!')
            threading.Thread(target=self.publish_ready, args=()).start()
        elif parameters[0] == 'UpdateValues':
            threading.Thread(target=self.client.modify_values, args=()).start()
        elif (parameters[0] == SERVER_BASE_TOPIC 
              and parameters[1] == SERVER_READY_TOPIC 
              and parameters[2] == self.cppu_name 
              and int(payload) == 1):
            threading.Thread(target= self.check_server_ready, args=[payload]).start()

    def update_training_regime(self, payload):
        """Update Trainig Regime with code
        0: self.train_mode = True
        1: self.inference_mode = True
        2: self.inference_mode = False & self.train_mode = False    
        """
        if payload == 0:
            self.train_mode = True
            self.inference_mode = False
            self.baseline_mode = False
        elif payload == 1:
            self.train_mode = False
            self.inference_mode = True
            self.baseline_mode = False
        else:
            self.inference_mode = False
            self.train_mode = False
            self.baseline_mode = True
        self.logger.info(f'Updated Training Regime to {payload}')
        
    def check_server_ready(self, payload):
        """ Method that waits from the MQTT message from the Rllib server to be ready"""
        self.logger.info(f'Agent received Server Ready message : {payload} ...') 
        self.server_ready.wait()

    def publish_ready(self):
        """Method used to publish to the orchestrator the confirmation that the Agent is ready"""
        mqtt_topic = '/'.join(['AgentReady', self.cppu_name])
        payload =  '1'
        self.client_ready.wait()
        self.logger.info(f'MQTT: sending {mqtt_topic} payload: {payload} ...') 
        with self.mqtt_pub_client_lock:
            self.mqtt_pub_client.publish(mqtt_topic, payload = payload)

    def publish_production_threshold(self):
        """Method used to publish the reach of a threshold to the orchestrator so as to stop the production and the episode"""
        mqtt_topic = '/'.join(['AI', 'DetectedThreshold', self.cppu_name])
        payload = self.inference_mode
        with self.mqtt_pub_client_lock:
            self.mqtt_pub_client.publish(mqtt_topic, payload = payload)

    def publish_port(self, port):
        """Method used to publish the port selected by the Agent to the Simulator"""
        exchange_dict = self.exchange_dicts['path']
        exchange_dict['port'] = port

        mqtt_topic = '/'.join(['PathSelection',
                               'cppc',
                               self.cppu_name])
        with self.mqtt_pub_client_lock:
            self.mqtt_pub_client.publish(mqtt_topic,
                                         json.dumps({"port": port}))
        self.logger.info(f'Port {port}')

    def publish_variant(self, cppu, variant, id_mqtt):
        """Method used to publish the variat chose by the agent"""
        exchange_dict = self.exchange_dicts['variant'][id_mqtt]
        exchange_dict['variant'] = variant
        mqtt_topic = '/'.join(['VariantSelection',
                               exchange_dict['cppu'],
                               exchange_dict['skill'],
                               exchange_dict['identifier']])
        with self.mqtt_pub_client_lock:
            self.mqtt_pub_client.publish(mqtt_topic, variant)
        self.logger.debug(f'MQTT[{id_mqtt}] VARIANT [{variant}] ' +
                          f'SKILL {exchange_dict["skill"]}')
        self.exchange_dicts['variant'].pop(id_mqtt)
        self.logger.debug(f'MQTT[{id_mqtt}] VARIANT Exchange Dict Deleted ...')

    def publish_path_learning_status(self, cppu):
        mqtt_topic = '/'.join(['PathStatus', cppu])
        with self.mqtt_pub_client_lock:
            self.mqtt_pub_client.publish(mqtt_topic, 'done')
        self.logger.debug(f'sent PathStatus done for {mqtt_topic}')

    def publish_variant_learning_status(self, cppu, skill, identifier):
        mqtt_topic = '/'.join(['VariantStatus', cppu, skill, identifier])
        with self.mqtt_pub_client_lock:
            self.mqtt_pub_client.publish(mqtt_topic, 'done')
        self.logger.debug(
            f'sent VariantStatus done for {mqtt_topic}')

    def start_episode(self, episode_id):
        """Start a new episode when receiving the MQTT message EpisodeManagement start"""
        self.reset()
        self.episode_identifier = episode_id
        if self.client_server:
            self.logger.debug(f'Starting episode {self.episode_identifier}')
            if self.update_interval is None and not self.off_policy:
                try:
                    self.client.update_policy_weights()
                except Exception as e:
                    self.logger.error(f'Failed to get policy from updated client {e}')
            self.client.start_episode(episode_id, training_enabled=self.train_mode)
            self.logger.debug('Start_episode has returned')
            self.publish_started_episode(episode_id)
        else:
            raise ValueError("Episode only started for client server")
            # if episode_id == 'Evaluation':
            #     self.logger.debug(f'Loading model for evaluation, out_dir={self.out_dir}')
            #     self.client.load(self.out_dir)

    def publish_started_episode(self, episode_id):
        """Method used to publish to the orchestrator the fact that the Agent started an episode"""
        mqtt_topic = '/'.join(['StartedEpisode', self.cppu_name, episode_id])
        payload = '1'
        self.logger.debug(f'MQTT: sending {mqtt_topic} payload: {payload} ...') 
        with self.mqtt_pub_client_lock:
            self.mqtt_pub_client.publish(mqtt_topic, payload = payload)
            self.logger.debug(f'Message {mqtt_topic} payload: {payload} SENT!')

    def end_episode(self, episode_id, produced_product, save_checkpoint, last_reward):
        """End the episode when receiving the MQTT message EpisodeManagement end"""
        if self.client_server:
            if save_checkpoint:
                try:
                    self.client.update_policy_weights()
                    agent_policy_trained = self.client.rollout_worker.get_policy()
                    self.logger.info(f'Policy Updated {agent_policy_trained}')
                    ckp_string = f"{self.out_dir}/models/{self.cppu_name}_{self.algorithm_class}_{episode_id}/"
                    os.makedirs(ckp_string, exist_ok=True)
                    self.logger.info(f'Exporting Policy Updated @ {ckp_string}')
                    agent_policy_trained.export_checkpoint(export_dir=ckp_string)
                    self.logger.info(f'Exported Policy Updated @ {ckp_string}')
                except Exception as e:
                    self.logger.error(f'Failed to get/save policy from updated client: {e}')
            
            if self.last_observation is None:
                assert last_reward == None
                # Dum transition
                dummy_observation = self.observation_space.sample()
                _ = self.client.get_action(episode_id, dummy_observation)
                dummy_reward = self.dummy_reward
                self.client.log_returns(episode_id, dummy_reward)
                self.logger.info(f'Dummy End Episode for #E{episode_id}')
                self.client.end_episode(episode_id, dummy_observation)
                self.logger.info(f'Ended Dummy End Episode for #E{episode_id}')
            else:
                # Log just the last reward
                last_sample_key = list(self.trajectory.keys())[-1]
                last_sample = self.trajectory[last_sample_key]
                assert episode_id == last_sample['info']['eid']
                # last_reward = self.compute_reward_rllib(produced_product['States']['SkillHistory'])
                self.client.log_returns(episode_id=episode_id, reward=last_reward)
                self.logger.info(f'End Episode for #E{episode_id}')
                self.client.end_episode(episode_id, self.last_observation)
                self.logger.info(f'Ended End Episode for #E{episode_id}')
            self.publish_ended_episode(episode_id)
        else:
            self.client.end_episode(episode_id, produced_product, save_checkpoint, self.out_dir)
        
        self.trajectory = {}
        self.transported_products = set()
        self.skill_histories = {}

    def publish_ended_episode(self, episode_id):
        """Method used to publish to the orchestrator the fact that the Agent ended an episode"""
        mqtt_topic = '/'.join(['EndedEpisode', self.cppu_name, episode_id])
        payload = '1'
        self.logger.info(f'MQTT: sending {mqtt_topic} payload: {payload} ...') 
        with self.mqtt_pub_client_lock:
            self.mqtt_pub_client.publish(mqtt_topic, payload = payload)
            self.logger.info(f'Message {mqtt_topic} payload: {payload} SENT!')
    


class Orchestrator(AgentSImulator):
        
        def __init__(self, cppu_name = '', learning_config = None, simulator_config = None,
                     product_config = None, mqtt_host_url=None, gateway_url=None, out_dir='./',
                     original_seed = None):

            self.logger = setup_logger(f'Agent::Orchestrator::PID{os.getpid()}')
            self.logger.debug(f'Setting up AgentSImulator interfacing')
            super().__init__(cppu_name, 
                             learning_config=learning_config,
                             config=simulator_config, 
                             products=product_config, 
                             logger=self.logger,
                             mqtt_host_url=mqtt_host_url, 
                             gateway_url=gateway_url)
            self.cppu_name = cppu_name
            self.learning_config = learning_config
            self.training = self.learning_config['training']
            self.episodes = self.learning_config['episodes']
            self.client_server = self.learning_config["client_server"]
            self.algorithm_class = self.learning_config['algorithm_class']
            self.checkpoint_frequency = self.learning_config['checkpoint_frequency']
            self.use_baseline = self.learning_config['use_baseline']
            self.demo_evaluation = self.learning_config['demo_evaluation']
            self.n_runs = self.learning_config['n_runs']
            self.out_dir = out_dir
            evaluation_info = dict()
            evaluation_info['learning_config'] = learning_config
            self.cppu_threshold_detected = None
            self.cppu_to_wait = []
            self.cppu_to_wait_lock = threading.Lock()
            self.cppu_ready = []
            self.cppu_ready_lock = threading.Lock()
            self.mqtt_pub_client = mqtt.Client()
            self.mqtt_pub_client_lock = threading.Lock()
            self.mqtt_sub_client = mqtt.Client()
            self.setup_mqtt_clients()
            self.wait_for_agents()

            logger.info(f'Original seed was {original_seed}')
            evaluation_info['seed'] = original_seed

            # LAUNCHING PHASES
            with open(os.path.join(out_dir, 'skill_history.txt'), 'w') as sk_file:
                sk_file.write('|SKILL HISTORY|\n')
                sk_file.write('|_____________|\n\n')

            # Baseline
            evaluation_info_before = dict()
            if self.use_baseline:
                raise ValueError("No baseline implementation yet")
                self.publish_training_mode(training=False, inference_mode=False)
                start_time = time.time()
                evaluation_info_before = self.evaluate_products(inference_mode=False)
                end_time = time.time()
                logger.info('BASELINE -- Elapsed time: {} s'.format(round(end_time-start_time, 2)))
                evaluation_info['baseline'] = evaluation_info_before

            # Training
            training_info = dict()
            if self.training:
                start_time = time.time()
                training_info, _ = self.train(self.training, self.episodes, self.checkpoint_frequency)
                end_time = time.time()
                logger.info('TRAINING -- Elapsed time: {} s'.format(round(end_time-start_time, 2)))
                evaluation_info['training'] = training_info

            # Evaluation (RL or Demo)
            self.reset_simulator()
            self.publish_training_mode(training=False, inference_mode=True)
            self.publish_episode_management(phase='Start', episode_id="Evaluation")
            self.sync_episode()
            
            # RL Testing
            evaluation_info_after = dict()
            if not self.demo_evaluation:
                self.logger.debug('sync_episode (start) has returned')
                start_time = time.time()
                evaluation_info_after = self.evaluate_products(inference_mode=True)
                end_time = time.time()
                logger.info('EVALUATION -- Elapsed time: {} s'.format(round(end_time-start_time, 2)))
                evaluation_info['evaluation'] = evaluation_info_after
                self.publish_episode_management(phase='End', 
                                                episode_id = "Evaluation")
                self.sync_episode()
                self.logger.debug('sync_episode (end) has returned')
            
            logger.info('Saving extended_info.json')

            with open(os.path.join(out_dir, 'extended_info.json'), "w") as outfile: 
                json.dump(evaluation_info, outfile, indent=6)
            # Building a succint output
            evaluation_file = self.build_evaluation_output_file(learning_config, original_seed, evaluation_info)

            logger.info('Saving evaluation_info.json')
            with open(os.path.join(out_dir, 'evaluation_info.json'), "w") as outfile:
                json.dump(evaluation_file, outfile, indent=6)
            
        def sync_episode(self):
            """Assert all the agents are synchronized on the same episode"""
            if self.learning_config["client_server"]:
                while len(self.cppu_ready) < len(self.real_cppu_names):
                    continue
                self.cppu_ready = []
            else:
                assert True

        def build_evaluation_output_file(self, l_config, seed, evaluation_info):
            # Building an evaluation file with only necessary information
            evaluation_file = dict()
            evaluation_file["learning_config"] = l_config
            evaluation_file["seed"] = seed
            for phase, phase_data in evaluation_info.items():
                if phase == "baseline" and self.use_baseline:
                    evaluation_file["baseline"] = self.get_info_non_training(phase_data)
                elif phase == "training":
                    evaluation_file["training"] = self.get_info_training(phase_data, include_path=False)
                elif phase == "evaluation":
                    evaluation_file["evaluation"] = self.get_info_non_training(phase_data)
            return evaluation_file

            
        def wait_for_agents(self):
            self.cppu_to_wait = []
            for cppu_name in self.real_cppu_names:
                self.cppu_to_wait.append(cppu_name)
            for cppu_name in self.cppu_to_wait:
                with self.cppu_to_wait_lock:
                    self.logger.debug(f'The agents to wait are: {self.cppu_to_wait}')
                    # Send the message to every agent
                    mqtt_topic = '/'.join([ 'AgentReadyRequest', cppu_name])
                    with self.mqtt_pub_client_lock:
                        payload = '1'
                        retain = True
                        self.logger.debug(f'MQTT: sending {mqtt_topic} ' +
                                          f'with payload {payload} and retain {retain}')
                        self.mqtt_pub_client.publish(mqtt_topic,
                                                     payload=payload,
                                                     retain=retain)
                        self.logger.debug(f'MQTT: sent {mqtt_topic} ' +
                                          f'with payload {payload} and retain {retain}')
            self.logger.debug(f'Waiting ...')
            # If there is an agent not ready yet
            while len(self.cppu_to_wait) > 0:
               continue
            self.logger.info(f'All the agents are ready ...')
            payload = '0'
            retain = True
            for cppu_name in self.real_cppu_names:
                with self.mqtt_pub_client_lock:
                    mqtt_topic = '/'.join(['AgentReadyRequest', cppu_name])
                    self.logger.debug(f'MQTT: sending {mqtt_topic} ' +
                                      f'with payload {payload} and retain {retain}')
                    self.mqtt_pub_client.publish(mqtt_topic, payload,retain=retain)
                    self.logger.debug(f'MQTT: sent {mqtt_topic} ' +
                                      f'with payload {payload} and retain {retain}')
            
            with self.mqtt_pub_client_lock:
                mqtt_topic = '/'.join([ 'AgentReadyRequest', self.cppu_name])
                self.logger.debug(f'MQTT: sending {mqtt_topic} ' +
                                  f'with payload {payload} and retain {retain}')
                self.mqtt_pub_client.publish(mqtt_topic, payload, retain = retain)
                self.logger.debug(f'MQTT: sent {mqtt_topic} ' +
                                  f'with payload {payload} and retain {retain}')

        def setup_mqtt_clients(self):
            self.logger.debug(f'MQTT: setting client @ {self.get_mqtt_hostname()}...')
            for mqtt_client in (self.mqtt_pub_client, self.mqtt_sub_client):
                mqtt_client.on_connect = self.on_connect
                mqtt_client.connect(self.get_mqtt_hostname())
            self.logger.debug(f'MQTT client set up ...')
            self.mqtt_sub_client.on_message = self.mqtt_sub_callback
            self.logger.debug(f'Subscribing to AgentReady')
            self.mqtt_sub_client.subscribe(f'AgentReady/#')
            self.mqtt_sub_client.subscribe(f'StartedEpisode/#')
            self.mqtt_sub_client.subscribe(f'EndedEpisode/#')
            self.mqtt_sub_client.subscribe(f'AI/#')
            self.mqtt_sub_client.loop_start()
            self.mqtt_pub_client.loop_start()
            
        def on_connect(self, client, userdata, flags, rc):
            self.logger.debug(f'MQTT: connection status {rc}')

        def mqtt_sub_callback(self, client, userdata, message):
            parameters = message.topic.split('/')
            payload = message.payload.decode('utf-8')
            self.logger.debug(f'Received {parameters}')
            if parameters[0] == 'AgentReady' and int(payload) == 1:
                self.logger.debug(f'MQTT: {parameters} with payload {int(payload)}')
                threading.Thread(target=self.remove_cppus,
                                 args=((int(payload), *parameters[1:]))).start()
            elif parameters[0] == 'StartedEpisode' and int(payload) == 1:
                self.logger.debug(f'MQTT: {parameters} with payload {int(payload)}')
                threading.Thread(target=self.add_cppus,
                                 args=(int(payload), parameters[1])).start()
            elif parameters[0] == 'EndedEpisode' and int(payload) == 1:
                self.logger.debug(f'MQTT: {parameters} with payload {int(payload)}')
                threading.Thread(target=self.add_cppus,
                                 args=((int(payload), parameters[1]))).start()
            elif parameters[0] == 'AI' and parameters[1] == 'DetectedThreshold':
                self.logger.debug(f'MQTT: {parameters} with payload {payload}')
                threading.Thread(target=self.handle_production_threshold,
                                 args=((payload, parameters[2]))).start()

        def handle_production_threshold(self, payload, cppu_name):
            self.logger.info(f"Handling Production Threshold detected by {cppu_name} with inference_mode {payload}")
            self.cppu_threshold_detected = cppu_name
            if payload == "True":
                self.logger.info(f"Production Threshold in Evaluation, Failed Learning")
                self.logger.error(f"Production Threshold in Evaluation, Failed Learning")
                # raise RuntimeError("Production Threshold in Evaluation, Failed Learning")
            else:
                # start new episode
                mqtt_topic = 'AI/TrainingLoop'
                payload = '{"Text" : "Production Threshold"}'
                with self.mqtt_pub_client_lock:
                    self.mqtt_pub_client.publish(mqtt_topic, payload)

        def remove_cppus(self, payload, cppu_name):
            with self.cppu_to_wait_lock:
                self.cppu_to_wait.remove(cppu_name)

        def add_cppus(self, payload, cppu_name):
            with self.cppu_ready_lock:
                self.cppu_ready.append(cppu_name)
                self.logger.debug('Appended {} to cppu_ready: {}'.format(cppu_name, self.cppu_ready))

        def update_values_after_training(self):
            self.logger.debug('Sending MQTT message UpdateValues')
            mqtt_topic = '/'.join(['UpdateValues', self.cppu_name])
            payload = ''
            with self.mqtt_pub_client_lock:
                self.mqtt_pub_client.publish(mqtt_topic, payload)

        def train(self, training, episodes, checkpoint_frequency):
            """
            After resetting the simulator and loading all the possible products it, for each of it:
            (0) initialize episode for the client learner.debug
            (1) select one product
            (2) start a simulation for one product
            (3) wait for the simulation to finish
            (4) access the finished products
            (5) compute the total energy consumption of the product
            (6) compute the total production duration of the product
            (7) back-propagate rewards for the reward-assignment problem
            """
            products = self.get_product_names()
            self.publish_training_mode(training=True, inference_mode=True)
            # self.wait_for_agents()
            step_info = {}

            for episode in range(episodes):
                step_info[str(episode)] = {}
                episode_id = f"E{episode}"

                self.logger.info('Episode Management: START {}'.format(episode_id))
                self.publish_episode_management('Start', episode_id)
                self.sync_episode()
                self.logger.info('Episode Management: START {} SYNC OK'.format(episode_id))
            
                product_name = random.choice(products)
                self.logger.info(make_pretty_log_entry(
                    f'[TRAINING] Product {product_name} ({episode + 1}/{episodes}) ...'))
                # (2)
                augmented_product_name = self.start_produce_product(product_name)
                self.logger.info(f'The augmented product name is {augmented_product_name}')
                # (3)
                self.wait_until_product_is_produced(augmented_product_name)
                # (4)
                product_type, produced_product = self.delete_products_after_condition(augmented_product_name,
                                                                                      self.cppu_threshold_detected)
                # (5)
                energy_consumption = self.compute_energy_consumption(produced_product) if product_type is not None else None
                self.logger.info(f'[KPI] Energy Consumption: {energy_consumption}') if product_type is not None else None
                # (6)
                duration = self.compute_total_duration(produced_product, title='Episode {}'.format(episode+1))
                self.logger.info(f'[KPI] Total Duration: {duration}')
                # (7)
                step_info[str(episode)][product_type] = {
                    'energy_consumption': energy_consumption, 
                    'duration': duration, 
                    'product': produced_product}
                if checkpoint_frequency is None:
                    save_checkpoint = False
                else:
                    save_checkpoint = True if (episode+1) % checkpoint_frequency == 0 or episode == episodes-1 else False
                self.publish_episode_management('End', episode_id,
                                                produced_product=produced_product,
                                                save_checkpoint=save_checkpoint)
                self.logger.info('Episode Management: END {}'.format(episode_id))
                self.sync_episode()
                self.logger.info('Episode Management: END {} SYNC OK'.format(episode_id))

            if not self.client_server:
                self.update_values_after_training()

            # Return the training info
            return step_info, produced_product

        def evaluate_products(self, inference_mode):
            """
            # For all the producible parts of the simulator (0):
            # (1) start producing each of it   
            # (2) wait for the process to finish 
            # (3) finish producing them
            # (4, 5) computes energy consumption and total time duration of the process
            """
            products = self.get_product_names()
            info_dict = {}
            # self.wait_for_agents()

            for run in np.arange(self.n_runs):
                
                product_name = random.choice(products)
                if inference_mode:
                    self.logger.info(make_pretty_log_entry(f'[EVALUATION N {run}] product {product_name}'))
                else:
                    self.logger.info(make_pretty_log_entry(f'[BASELINE N {run}] product {product_name}'))
                # (1)
                augmented_product_name = self.start_produce_product(product_name)
                self.logger.info(f'The augmented product name is {augmented_product_name}')
                # (2)
                self.wait_until_product_is_produced(augmented_product_name)
                self.logger.info(f'The product has been produced')
                # (3)
                self.logger.info(f"CPPU {self.cppu_threshold_detected} detected threshold")
                product_type, produced_product = self.delete_products_after_condition(augmented_product_name, self.cppu_threshold_detected)
                self.logger.info(f'The product type is {product_type}')
                # (4)
                energy_consumption = self.compute_energy_consumption(produced_product) if product_type is not None else None
                duration = self.compute_total_duration(produced_product, title='Evaluation with agent scheduling = {}'.format(inference_mode)) if product_type is not None else None
                self.logger.info(f'[KPI] Energy Consumption {energy_consumption}')
                self.logger.info(f'[KPI] Total Duration: {duration}')
                info_dict[str(run)] = {'energy_consumption': energy_consumption,
                             'duration': duration,
                             'inference_mode': inference_mode,
                             'product': produced_product}
            return info_dict
            
        def publish_training_mode(self, training, inference_mode):
            """Managing Training"""
            if training:
                training_regime_code = 0
            elif inference_mode:
                training_regime_code = 1
            else:
                training_regime_code = 2

            for cppu_name in self.real_cppu_names:
                mqtt_topic = '/'.join(['TrainingRegime', cppu_name])
                with self.mqtt_pub_client_lock:
                    payload = str(training_regime_code)
                    self.logger.info(f'Sending a Training Regime Code {payload} to {cppu_name}')
                    self.mqtt_pub_client.publish(mqtt_topic, payload)
                self.logger.info(
                    f'Sent TrainingRegime: Training {training} ' +
                    f'inference_mode {inference_mode} to {cppu_name}')

        def publish_episode_management(self, phase, episode_id,
                                       produced_product=None,
                                       save_checkpoint=None):
            """Start or ending and episode"""
            payload = {}
            payload['episode_id'] = episode_id

            if phase == 'End':
                payload['produced_product'] = produced_product
                payload['save_checkpoint'] = save_checkpoint
            list_of_cppus = self.extract_list_of_cppus()
            for cppu_name in list_of_cppus:
                mqtt_topic = '/'.join(['EpisodeManagement', cppu_name, phase])
                with self.mqtt_pub_client_lock:
                    self.mqtt_pub_client.publish(mqtt_topic, json.dumps(payload))

        def extract_list_of_cppus(self):
            """This method, given a product, should return a list of all the
            cppus that joint the production"""
            return self.real_cppu_names

        def compute_energy_consumption(self, product):
            energy_consumption = 0
            if 'Components' in product:
                for component_product in product['Components'].values():
                    energy_consumption += self.compute_energy_consumption(component_product)

            skill_history = product['States']['SkillHistory']
            for skill_entry in skill_history:
                idle_energy_consumption = skill_entry['IdleEnergyConsumption']
                dynamic_energy_consumption = sum([behavior['DynamicEnergyConsumption']
                                                for behavior in skill_entry['Behaviors']])
                energy_consumption += idle_energy_consumption + dynamic_energy_consumption
            return energy_consumption

        def compute_total_duration(self, product, title):
            start_time = self.get_production_start_time(product)
            end_time = product['States']['SkillHistory'][-1]['EndTime']
            total_duration = end_time - start_time + 1

            logger.info('Saving SkillHistory')
            sk_file = open(os.path.join(self.out_dir, 'skill_history.txt'), 'a')
            sk_file.write('---------------------------------------------------\n')
            sk_file.write(title + '\n')
            sk_file.write('---------------------------------------------------\n')
            for d in product['States']['SkillHistory']:
                sk_file.write(d['Cppu'] + '\n')
                for b in d['Behaviors']:
                    sk_file.write('{} --> {}\n'.format(b['Behavior'], b['Duration']))
                sk_file.write('Total duration: {}\n'.format(d['Duration']))
                sk_file.write('----------------\n')
            sk_file.write('FINAL DURATION: {}\n\n'.format(total_duration))
            sk_file.close()
            return total_duration
        
        def get_production_start_time(self, product):
            # the start time of the product is the minimum of the start time
            # of the first skill of this product ...
            start_time = product['States']['SkillHistory'][0]['StartTime']
            if 'Components' in product:
                for component_product in product['Components'].values():
                    # ... and any start time of the first skill of any product component
                    start_time = min(start_time, self.get_production_start_time(component_product))
            # returns the minimum start time present in the product
            return start_time
        
        def get_all_products_histories(self, product_id, product_info_path):
            """Return a dictionary with keys the product components names,
            and as values their histories"""
            product_histories = {}

            if "States" in product_info_path.keys():
                product_histories[product_id] = {}
                product_histories[product_id]["States"] = product_info_path["States"]
                product_histories[product_id]["Subproducts"] = None
                product_histories[product_id]["Product"] = product_info_path["Configuration"]["Product"]

            if "Components" in product_info_path.keys():
                product_histories[product_id]["Subproducts"] = list(product_info_path["Components"].keys())
                for component in product_info_path["Components"].keys():
                    retrieved_histories = self.get_all_products_histories(
                        component, product_info_path["Components"][component])

                    for sub_component, sub_history in retrieved_histories.items():
                        product_histories[sub_component] = {}
                        product_histories[sub_component]["States"] = sub_history["States"]
                        product_histories[sub_component]["Subproducts"] = sub_history["Subproducts"]
                        product_histories[sub_component]["Product"] = sub_history["Product"]

            return product_histories
        
        def get_start_time(self, elem):
            """Return the start time of the entry if it is in the right format"""
            return elem['start_time'] if 'start_time' in elem.keys() else np.inf
        
        def get_info_non_training(self, info):
            """Retrieve the baseline data"""
            info_output = {}
            for run, info_run in info.items():
                info_output[str(run)] = {}
                product = info_run['product']
                product_id = 'main_product'

                info_output[str(run)]['duration'] = info_run['duration']
                info_output[str(run)]['energy_consumption'] = info_run['energy_consumption']

                info_output[str(run)]['baseline_path'] = []
                product_histories = self.get_all_products_histories(product_id, product)
                for product_id, product_info in product_histories.items():
                    for skill_history in product_info["States"]["SkillHistory"]:
                        skill = skill_history['Skill']
                        cppu = skill_history['Cppu']
                        start_time = skill_history['StartTime']

                        info_output[str(run)]['baseline_path'].append({
                            'product': product_info['Product'],
                            'cppu': cppu,
                            'skill': skill,
                            'start_time': start_time,
                            'product_id': product_id
                        })

                info_output[str(run)]['baseline_path'].sort(key=self.get_start_time)
            return info_output

        def get_info_training(self, all_info_training, include_path = False):
            """Retrieve the training data"""
            new_info_training = {}

            for episode, info_training in all_info_training.items():
                new_info_training[episode] = {}

                if include_path:
                    new_info_training[episode]['training_path'] = []

                for product_name, product_data in all_info_training[episode].items():

                    new_info_training[episode]['duration'] = product_data['duration']
                    new_info_training[episode]['energy_consumption'] = product_data['energy_consumption']

                    if include_path:
                        product_histories = self.get_all_products_histories(
                            product_name, product_data['product'])

                        for product_id, product_info in product_histories.items():
                            for skill_history in product_info["States"]["SkillHistory"]:
                                skill = skill_history['Skill']
                                cppu = skill_history['Cppu']
                                start_time = skill_history['StartTime']

                                new_info_training[episode]['training_path'].append({
                                    'product': product_info['Product'],
                                    'cppu': cppu,
                                    'skill': skill,
                                    'start_time': start_time,
                                    'subproducts': product_info['Subproducts'],
                                    'product_id': product_id
                                })

                if include_path:
                    new_info_training[episode]['training_path'].sort(key=self.get_start_time)

            return new_info_training



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--orchestrator')
    parser.add_argument('--cppu_name')
    parser.add_argument('--learning_config')
    parser.add_argument('--learning_config_path')
    parser.add_argument('--simulator_config')
    parser.add_argument('--simulator_config_path')
    parser.add_argument('--product_config')
    parser.add_argument('--product_config_path')
    parser.add_argument('--out_dir')
    parser.add_argument('--serverport')
    parser.add_argument('--seed')

    mqtt_host_url = os.getenv('MQTT_HOST_URL')
    gateway_url = os.getenv('CPPS_GATEWAY_URL')

    try:
        args = parser.parse_args()
        logger = setup_logger(f'AgentLaunching::{args.cppu_name}')
        if args.learning_config is not None:
            learning_config = args.learning_config
        else:
            with open(args.learning_config_path) as config_file:
                learning_config = json.load(config_file)

        if args.simulator_config is not None:
            simulator_config = args.simulator_config
        else:
            with open(args.simulator_config_path) as config_file:
                simulator_config = json.load(config_file)

        if args.product_config is not None:
            product_config = args.product_config
        else:
            with open(args.product_config_path) as config_file:
                product_config = json.load(config_file)

        if args.orchestrator is not None:
            logger.info('Launching Orchestrator (blocking)')
            agent = Orchestrator(cppu_name=args.cppu_name,
                                 learning_config=learning_config,
                                 simulator_config=simulator_config,
                                 product_config=product_config,
                                 out_dir=args.out_dir,
                                 mqtt_host_url=mqtt_host_url,
                                 gateway_url=gateway_url,
                                 original_seed=args.seed)
            agent.mqtt_host_url = mqtt_host_url
        else:
            logger.debug('Creating Agent')
            serverport = 0
            if args.serverport:
                serverport = args.serverport
            agent = Agent(cppu_name=args.cppu_name,
                          learning_config = learning_config,
                          simulator_config = simulator_config,
                          product_config = product_config,
                          mqtt_host_url=mqtt_host_url,
                          gateway_url=gateway_url,
                          serverport=serverport,
                          seed=args.seed,
                          out_dir=args.out_dir)
            logger.info('Waiting for the Agent to finish (forever)')
            while True:
                time.sleep(1)
    except Exception as e:
        logger.error(f"Fatal exception during the agent/orchestrator startup:")
        logger.error(e, exc_info=True)
        raise

