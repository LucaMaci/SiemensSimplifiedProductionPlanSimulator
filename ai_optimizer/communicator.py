import paho.mqtt.client as mqtt
from typing import Literal
import json
import threading


class Communicator:
    def __init__(
            self,
            n_agents: int = 9,
            mqtt_hostname: str = None,
    ):
        self.n_agents = n_agents
        self.mqtt_hostname = mqtt_hostname
        self.cppu_names = [f'cppu_{i}' for i in range(n_agents)]
        """
        Class handling the communications
        """

        self.actions = {}
        # Setting up barriers for MQTT sync
        self.reward_barriers = {'variant': {}, 'path': {}}
        self.variant_history_dict_lock = threading.Lock()
        self.path_history_dict_lock = threading.Lock()
        self.server_ready = threading.Barrier(2)
        self.client_ready = threading.Barrier(2)
        self.action_barriers = {}
        self.cppu_to_wait = []
        self.cppu_to_wait_lock = threading.Lock()
        self.cppu_ready = []
        self.cppu_ready_lock = threading.Lock()
        self.mqtt_setup()
        print("Waiting until the agents are up!")
        self.wait_for_agents()
        self.publish_training_mode(training=True, inference_mode=True)

    def mqtt_setup(self):
        self.mqtt_pub_client = mqtt.Client()
        self.mqtt_sub_client = mqtt.Client()
        for mqtt_client in (self.mqtt_pub_client, self.mqtt_sub_client):
            mqtt_client.connect(self.mqtt_hostname)
        self.mqtt_sub_client.on_message = self.mqtt_sub_callback
        self.mqtt_sub_client.subscribe(f'PathSelection/#')
        self.mqtt_sub_client.subscribe(f'AgentReady/#')
        self.mqtt_sub_client.subscribe(f'StartedEpisode/#')
        self.mqtt_sub_client.subscribe(f'EndedEpisode/#')
        self.mqtt_pub_client_lock = threading.Lock()
        self.mqtt_sub_client.loop_start()
        self.mqtt_pub_client.loop_start()

    def publish_training_mode(self, training, inference_mode):
        """Managing Training"""
        if training:
            training_regime_code = 0
        elif inference_mode:
            training_regime_code = 1
        else:
            training_regime_code = 2
        for cppu_name in self.cppu_names:
            mqtt_topic = '/'.join(['TrainingRegime', cppu_name])
            with self.mqtt_pub_client_lock:
                payload = str(training_regime_code)
                self.mqtt_pub_client.publish(mqtt_topic, payload)

    def wait_for_agents(self):
        self.cppu_to_wait = []
        for cppu_name in self.cppu_names:
            self.cppu_to_wait.append(cppu_name)
        for cppu_name in self.cppu_to_wait:
            with self.cppu_to_wait_lock:
                # Send the message to every agent
                mqtt_topic = '/'.join(['AgentReadyRequest', cppu_name])
                with self.mqtt_pub_client_lock:
                    payload = '1'
                    retain = True
                    self.mqtt_pub_client.publish(mqtt_topic,
                                                 payload=payload,
                                                 retain=retain)
        # If there is an agent not ready yet
        while len(self.cppu_to_wait) > 0:
            continue
        payload = '0'
        retain = True
        for cppu_name in self.cppu_names:
            with self.mqtt_pub_client_lock:
                mqtt_topic = '/'.join(['AgentReadyRequest', cppu_name])
                self.mqtt_pub_client.publish(mqtt_topic, payload, retain=retain)

        # with self.mqtt_pub_client_lock:
        #     mqtt_topic = '/'.join(['AgentReadyRequest', self.cppu_name])
        #     self.mqtt_pub_client.publish(mqtt_topic, payload, retain=retain)

    def mqtt_sub_callback(self, client, userdata, message):
        parameters = message.topic.split('/')
        payload = message.payload.decode('utf-8')
        cppu_name = parameters[1]

        if parameters[0] == 'PathSelection' and cppu_name == 'cppc':
            cppu_name = parameters[2]
            payload = json.loads(payload)
            self.actions[cppu_name] = payload["port"]
            self.action_barriers[cppu_name].wait()
        elif parameters[0] == 'AgentReady' and int(payload) == 1:
            threading.Thread(target=self.remove_cppus,
                             args=((int(payload), *parameters[1:]))).start()
        elif parameters[0] == 'StartedEpisode' and int(payload) == 1:
            threading.Thread(target=self.add_cppus,
                             args=(int(payload), parameters[1])).start()
        elif parameters[0] == 'EndedEpisode' and int(payload) == 1:
            threading.Thread(target=self.add_cppus,
                             args=((int(payload), parameters[1]))).start()

    def send_state_and_previous_reward(self, cppu, state, prev_reward, threshold_detected=False):
        # exchange_dict = {"cppu": cppu}
        mqtt_topic = '/'.join(['PathSelection', cppu])
        with self.mqtt_pub_client_lock:
            self.mqtt_pub_client.publish(mqtt_topic,
                                         json.dumps({"state": state,
                                                     "reward": prev_reward,
                                                     "threshold_detected": threshold_detected}))
        return f'State {state}'

    def send_transition(self, cppu, state, action, reward, next_state, done):
        mqtt_topic = '/'.join(['PathState', cppu])
        with self.mqtt_pub_client_lock:
            self.mqtt_pub_client.publish(mqtt_topic,
                                         json.dumps({"state": state,
                                                     "action": action,
                                                     "reward": reward,
                                                     "next_state": next_state,
                                                     "done": done}))
        return f'Sent Transition'

    def set_action_barrier(self, agent):
        self.action_barriers[agent] = threading.Barrier(2)

    def receive_action(self, agent):
        self.action_barriers[agent].wait()
        action = self.actions[agent]
        return action

    def sync_episode(self):
        """Assert all the agents are synchronized on the same episode"""
        while len(self.cppu_ready) < len(self.cppu_names):
            continue
        self.cppu_ready = []

    def remove_cppus(self, payload, cppu_name):
        with self.cppu_to_wait_lock:
            self.cppu_to_wait.remove(cppu_name)

    def add_cppus(self, payload, cppu_name):
        with self.cppu_ready_lock:
            self.cppu_ready.append(cppu_name)

    def publish_episode_management(self, phase, episode_id,
                                   last_rewards=None,
                                   produced_product=None,
                                   save_checkpoint=None):
        """Start or ending and episode"""
        payload = {}
        payload['episode_id'] = episode_id

        if phase == 'End':
            payload['produced_product'] = produced_product
            payload['save_checkpoint'] = save_checkpoint
        list_of_cppus = self.cppu_names
        for cppu_name in list_of_cppus:
            if phase == 'End':
                payload['last_reward'] = last_rewards[cppu_name]
            mqtt_topic = '/'.join(['EpisodeManagement', cppu_name, phase])
            with self.mqtt_pub_client_lock:
                self.mqtt_pub_client.publish(mqtt_topic, json.dumps(payload))