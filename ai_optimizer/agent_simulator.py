import os
import json
from typing import Iterable, List, Tuple, Union
import urllib
import functools
import multiprocessing
import numpy as np
import paho.mqtt.client as mqtt
from custom_policies import get_string_observation
import gymnasium as gym


ENV_SIMULATOR = 'CPPS_SIMULATOR_HOME'
# TODO FIX MAX SKILL TO BE COMPUTED
MAX_SKILL = 2


def getvar(varname, val, default=''):
    if val is not None:
        os.environ[varname] = val
    elif varname not in os.environ:
        os.environ[varname] = default
    return os.environ[varname]

def mqtt_callback(client, userdata, message, mqtt_client, augmented_product_name):
    message = json.loads(message.payload.decode('utf-8'))
    if 'Severity' in message.keys():
        if (message['Severity'] == 'success' and
            message['Text'] == f'Production of {augmented_product_name} completed.'):
            mqtt_client.disconnect()
    elif 'Text' in message.keys(): 
        if message['Text'] == f'Production Threshold':
            mqtt_client.disconnect()

def binary_list_to_int(l):
    res = 0
    for b in l:
        res = (res << 1) | b
    return res



class AgentSImulator():
    def __init__(self, cppu_name, learning_config=None, config=None, products=None, logger=None, mqtt_host_url=None,
                 gateway_url=None):
        self.logger = logger
        self.mqtt_host_url = mqtt_host_url \
            if mqtt_host_url is not None else 'localhost'

        if config is not None:
            self.logger.debug(f"Reading config from input")
            self.config = config
        else:
            self.logger.debug(f"Loading config from folder")
            self.config = self.__load_from_folder('config.json')

        if products is not None:
            self.logger.debug(f"Reading products from input")
            self.products = products
        else:
            self.logger.debug(f"Loading products from folder")
            self.products = self.__load_from_folder('products.json')
        
        # set the CPPU names used in the simulator including the plant
        self.n_agents = self.config["n_agents"]
        self.all_cppu_names = [f'cppu_{i}' for i in range(self.n_agents)]

        # set the CPPU names used in the simulator that are not a plant
        self.real_cppu_names = [f'cppu_{i}' for i in range(self.n_agents)]
        self.cppc_name = "" # TODO don't know if needed
        self.cppu_dictionary = {real_cppu_name: i-1 for i, real_cppu_name in enumerate(self.real_cppu_names, 1)}
        self.cppu_name = cppu_name
        self.index = cppu_name.split("_")[-1]
        if self.cppu_name in self.real_cppu_names:
            self.ports_info = None # TODO don't know if needed
            self.ports = [i for i, value in enumerate(config["agents_connections"][self.index]) if value is not None]
            # self.ports = [port for port in self.get_ports(cppu_name) if
            #               self.get_port_simulator(cppu_name, port) ['Configuration']['Connected'] is not None]
            self.skills = self.get_skills(self.cppu_name)
            self.skill_durations = self.config["agents_skills_custom_duration"]

        self.product_names = [f'product_{i}' for i in range(self.config["n_products"])]
        self.skill_names = [f'skill_{i}' for i in range(self.config["n_production_skills"])]

        # self.product_names = self.get_subproducts(self.product_names, self.products)
        # self.product_names.sort()
        # self.skill_names = list(self.config["behaviors"]["factors"]["skills"].keys())
        # self.skill_names.sort()

        # self.product_skills_pair = self.get_skills_per_product(self.products)
        
        # Learning-realted quantities
        self.learning_config = learning_config
        self.one_hot_state = self.learning_config['one_hot_state']
        self.shaping_value = self.learning_config['shaping_value']
        self.fact_duration = self.learning_config['fact_duration']
        self.fact_energy = self.learning_config['fact_energy']
        self.use_masking = self.learning_config['use_masking']

        self.observation_space_dict = {key: value for (key, value) in
                                       self.learning_config["observation_space_dict"].items() if value}
        self.action_space = gym.spaces.Discrete(len(self.ports))
        self.non_production_skill = ["transport", "defer", "buffer"]
        self.counter_dict = {}

        # create a mutex for the good quality storage
        self.storage_lock = multiprocessing.Lock()

    def __load_from_folder(filename: str) -> dict:
        """Load simulator config/products from specific folder"""
        path_here = os.path.realpath(os.path.dirname(__file__))
        home = getvar(ENV_SIMULATOR, None, path_here)
        config_home = os.path.join(home, 'configs')
        path_config = os.path.join(config_home, filename)
        with open(path_config) as config_file:
            return json.load(config_file)
        
    def get_skills_per_product(self, products):
        """ Return a dictionary of the skills needed by each product"""
        skills_per_product = self.get_product_skills_pairs(products.keys(), products)
        temp_dict = {}
        for pair in skills_per_product:
            temp_dict[pair[0]] = list(pair[1])
        # To save all the dictionary with the same order (across multiple cppu)
        sorted_names = sorted(temp_dict.keys())
        product_skills_pair = {}
        for key in sorted_names:
            product_skills_pair[key] = temp_dict[key]

        return product_skills_pair

    def get_table_products(self):
        "Return all the products in all the CPPUs"
        product_table = {}
        for cppu in self.real_cppu_names:
            product_table[cppu] = self.get_products(cppu)
        return product_table

    def get_port(self, port_index):
        """Given a selected action(a port index), 
        return the corresponding port string"""
        return self.ports[port_index]
    
    def get_next_cppu_name(self, next_port):
        cppu_config = self.config['plant_layout'][self.cppu_name]
        for connection in cppu_config['connections']:
               if connection['LocalPort'] == next_port:
                    cppu_name_next = connection['RemoteEquipment']
        return cppu_name_next
        
    def get_subproducts(self, product_names, config_path):
        """
        product_names are the name extracted from the config
        config_path is the portion of the Json where the product_names are the keys
        """
        products_to_return = set()

        for product_name in product_names:
            products_to_return.add(product_name)
            sub_products = config_path[product_name]["Components"].keys()
            retrieved_sub_products = self.get_subproducts(sub_products,
                                                          config_path[product_name]["Components"])
            for retrieved_sub_product in retrieved_sub_products:
                products_to_return.add(retrieved_sub_product)

        return list(products_to_return)
    
    def get_product_skills_pairs(self, product_list, products_config):
        """Get all the skills for each product"""
        product_skills_pair = set()

        for product in product_list:
            skills_list = products_config[product]['Configuration']['BOP']
            skills_to_add = []
            for skill_list in skills_list:
                skill = skill_list['Skill']
                skills_to_add.append(skill)

            product_skills_pair.add((product, tuple(skills_to_add)))

            subproducts_dict = products_config[product]['Components']
            subproducts_name = products_config[product]['Components'].keys()

            retrieved_pairs = self.get_product_skills_pairs(subproducts_name, subproducts_dict)

            for retrieved_pair in retrieved_pairs:
                product_skills_pair.add(retrieved_pair)

        return product_skills_pair
    

    def compute_reward(self, skill_history):
        """Compute reward as the negative sum of the duration of the last behavior
        executed by the agent (skills + transport)"""
        history_counter = 1
        production_kpi = 0
        overall_kpi = 0
        while history_counter <= len(skill_history):
            if skill_history[-history_counter]["Cppu"] == self.cppu_name:
                overall_kpi += self.extract_kpi(skill_history[-history_counter])
                if skill_history[-history_counter]["Skill"] not in self.non_production_skill:
                    production_kpi += self.extract_kpi(skill_history[-history_counter])
            else:
                break
            history_counter += 1
        self.logger.info(f"KPI base reward: {overall_kpi}")
        reshaped_reward = - self.shape_reward(overall_kpi=overall_kpi, production_kpi=production_kpi)
        self.logger.info(f"KPI reshaped reward: {reshaped_reward}")
        return reshaped_reward
    

    def shape_reward(self, overall_kpi, production_kpi):
        """
        Shape the reward
        - non_production_kpi: any kind of ,
        - skills_duration: is the duration due to execution skills (it's contained in computed_duration)
        """
        return overall_kpi - self.shaping_value * production_kpi

    def compute_reward_rllib(self, skill_history):
        """ Compute the reward in a Semi-MDP fashion for RLlib"""
        self.logger.debug(f"RLLib Computing KPI for skill history: {skill_history}")
        
        history_counter = 1
        production_kpi = 0
        overall_kpi = 0
        while history_counter <= len(skill_history):
            if skill_history[-history_counter]["Cppu"] != self.cppu_name:
                overall_kpi += self.extract_kpi(skill_history[-history_counter])
                if skill_history[-history_counter]["Skill"] not in self.non_production_skill:
                    production_kpi += self.extract_kpi(skill_history[-history_counter])
            else:
                break
            history_counter += 1
        self.logger.info(f"RLLib Computed reward: {overall_kpi}")
        reshaped_reward = - self.shape_reward(overall_kpi=overall_kpi, production_kpi=production_kpi)
        self.logger.info(f"RLLib Shaped reward: {reshaped_reward}")
        return reshaped_reward

    def get_action_space_ohe(self):
        obs_space_low = []
        obs_space_high = []
        if self.observation_space_dict.get("next_skill", False):
            obs_space_low.append(np.zeros(len(self.skill_names)))
            obs_space_high.append(np.ones(len(self.skill_names)))
        if self.observation_space_dict.get("product_name", False):
            obs_space_low.append(np.zeros(len(self.product_names)))
            obs_space_high.append(np.ones(len(self.product_names)))
        if self.observation_space_dict.get("cppu_state", False):
            obs_space_low.append(np.zeros(len(self.product_names)))
            obs_space_high.append(np.ones(len(self.product_names)))
        if self.observation_space_dict.get("next_skills", False):
            obs_space_low.append(np.zeros(len(self.skill_names)))
            obs_space_high.append(np.ones(len(self.skill_names)))
        if self.observation_space_dict.get("counter", False):
            obs_space_low.append(np.zeros(1))
            obs_space_high.append(np.ones(1))
        if self.observation_space_dict.get("previous_cppu", False):
            obs_space_low.append(np.zeros(len(self.cppu_dictionary)))
            obs_space_high.append(np.ones(len(self.cppu_dictionary)))
        obs_space_low = np.concatenate(obs_space_low)
        obs_space_high = np.concatenate(obs_space_high)
        return gym.spaces.Box(low=obs_space_low, high=obs_space_high)

    def get_observation_space(self):
        """ Return a list with the dimensionality of each component of the observation """
        if self.one_hot_state:
            space = self.get_action_space_ohe()
        else:
            obs_space = []
            if self.observation_space_dict.get("next_skill", False):
                obs_space.append(len(self.skill_names))
            if self.observation_space_dict.get("product_name", False):
                obs_space.append(len(self.product_names))
            if self.observation_space_dict.get("cppu_state", False):
                obs_space.append(2**len(self.product_names))  # one-hot
            if self.observation_space_dict.get("next_skills", False):
                obs_space.append(2**len(self.skill_names))  # one-hot
            if self.observation_space_dict.get("counter", False):
                obs_space.append(2)  # binary
            if self.observation_space_dict.get("previous_cppu", False):
                obs_space.append(len(self.cppu_dictionary))
            space = gym.spaces.MultiDiscrete(obs_space)

        if self.use_masking:
            space = gym.spaces.Dict(
                {
                    "observations": space,
                    "action_mask": gym.spaces.Box(0, 1, shape=(len(self.ports),))
                }
            )
        self.observation_space = space
        return space

    def prepare_observation(self, state):
        if self.use_masking:
            obs = dict(state)
        else:
            obs = np.array(state)
        return obs

    def get_action_space(self):
        return self.action_space
    
    def get_position_in_observation(self, field_name):
        """Given the field, retrive the position in the observation
        TODO: use an enum operator"""
        try:
            return list(self.observation_space_dict.keys()).index(field_name)
        except Exception as e:
            self.logger.info("{}: {} extraction from observation".format(e, field_name))
   
    def get_one_hot_products_encoding(self, products_name):
        """Get the one-hot encoding of the specified products"""
        basic_encoding = [0] * len(self.product_names)
        for product in products_name:
            product_idx = self.product_names.index(product)
            basic_encoding[product_idx] += 1
        return basic_encoding

    def get_skills_encoding_by_product(self, product_name):
        """Given the product, create the embedding with all the skills
            required for its production"""
        skills_to_encode = self.product_skills_pair[product_name]
        basic_encoding = [0] * len(self.skill_names)
        for skill in skills_to_encode:
            idx = self.skill_names.index(skill)
            basic_encoding[idx] += 1
        return basic_encoding

    def get_skills_encoding(self, skills_list):
        """Given a list of the skills, create the encoding"""
        basic_encoding = [0] * len(self.skill_names)
        for skill in skills_list:
            idx = self.skill_names.index(skill)
            basic_encoding[idx] += 1
        return basic_encoding

    def get_update_skill_encoding(self, product_name, skill_executed):
        """Get the embedding of the remaining skills to be executed (by difference)"""
        original_encoding = self.get_skills_encoding_by_product(product_name)
        skills_executed = self.get_skills_encoding(skill_executed)
        return [x - y if x >= y else 0 for x,y in zip(original_encoding, skills_executed)]

    def get_skills_history(self, product_info):
        """Return a list of the skills already executed on the product"""
        skills_executed = []
        if "States" in product_info.keys():
            product_skill_history = product_info["States"]["SkillHistory"]
            self.logger.debug(f'{self.cppu_name}: product skill history is {product_skill_history}')
            for skill_info in product_skill_history:
                if skill_info["Skill"] != "transport":
                    self.logger.debug(f'{self.cppu_name}: ' +
                                      f'skill already executed is {skill_info["Skill"]}')
                    skills_executed.append(skill_info["Skill"])

        self.logger.debug(f'{self.cppu_name}: skills_executed are {skills_executed}')
     
        return skills_executed
    
    def get_next_agent_observation(self, observation, port):

        next_cppu = self.get_next_cppu_name(port)
        counter_dict = self.get_whiteboard(next_cppu)['counter_dict']
        
        product_name = None
        next_skill = None
        next_skills = None
        counter_key_list = []
        if self.observation_space_dict.get('product_name', False):
            product_name = observation[self.get_position_in_observation('product_name')]
            counter_key_list.append(product_name)
        if self.observation_space_dict.get('next_skill', False):
            next_skill = observation[self.get_position_in_observation('next_skill')]
        if self.observation_space_dict.get('next_skills', False):
            next_skills = observation[self.get_position_in_observation('next_skills')]
            counter_key_list.append(next_skills)
        if self.observation_space_dict.get('previous_cppu', False):
            counter_key_list.append(self.cppu_name)

        counter_key = get_string_observation(counter_key_list)
        if counter_key in counter_dict.keys():
            count_per_product = counter_dict[counter_key]
        else:
            count_per_product = 0
        dcounter = 1 if count_per_product > 0 else 0
        # TODO: get from next agent for multi-product dist-Q compatibility
        products_encoding_per_cppu = [1]

        next_observation = []
        for field_name in self.observation_space_dict.keys():
            if field_name == "next_skill":
                next_observation.append(next_skill)
            elif field_name == "product_name":
                next_observation.append(product_name)
            elif field_name == "cppu_state":
                next_observation.append(products_encoding_per_cppu)
            elif field_name == "next_skills":
                next_observation.append(next_skills)
            elif field_name == "counter":
                next_observation.append(dcounter)
            elif field_name == "previous_cppu":
                next_observation.append(self.cppu_name)
        self.logger.info(f"Obs_next: {observation}, next_agent_obs: {next_observation}")
        next_observation = tuple(next_observation)
        return next_observation
    
    def update_counter_dict(self, observation):
        counter_key_obs = []
        
        if self.observation_space_dict.get('product_name', False):
            product_name = observation[self.get_position_in_observation('product_name')]
            counter_key_obs.append(product_name)
        if self.observation_space_dict.get('next_skills', False):
            next_skills = observation[self.get_position_in_observation('next_skills')]
            counter_key_obs.append(next_skills)
        if self.observation_space_dict.get('previous_cppu', False):
            previous_cppu = observation[self.get_position_in_observation('previous_cppu')]
            counter_key_obs.append(previous_cppu)

        counter_key = get_string_observation(counter_key_obs)
        if counter_key in self.counter_dict.keys():
            self.counter_dict[counter_key] += 1
        else:
            self.counter_dict[counter_key] = 1


    def get_mqtt_hostname(self):
        """return the hostname of the MQTT broker"""
        hostname = urllib.parse.urlparse(f'{self.mqtt_host_url}').hostname
        return hostname if hostname is not None else 'localhost'

    def get_skills(self, cppu_name):
        """return all skills supported by the passed CPPU"""
        return self.config["agents_skills"][cppu_name.split("_")[-1]]

    def get_port_simulator(self, cppu_name, port):
        """get all the information about a single port of a CPPU"""
        pass

    def get_ports(self, cppu_name):
        """get a list of all port names present at a CPPU"""
        pass
    
    def get_ports_deep(self, cppu_name):
        """get a dictionary containing all ports and all their information"""
        pass

    def get_product(self, cppu_name, product):
        """get the product information as dictionary (currently present at the passed cppu)"""
        pass

    def get_products(self, cppu_names: Union[List[str], str]) -> Iterable[Tuple[str, str]]:
        """ get a list of all product IDs currently present at a CPPU or list of CPPUs
            returns tuples of (product_id, cppu_name)"""
        pass
    
    def get_whiteboard(self, cppu_name):
        pass

    def set_whiteboard(self, cppu_name, payload):
        pass
    
    def create_digitaltwin_subscription(self, cppu_name, key):
        pass
    
    def wait_until_product_is_produced(self, augmented_product_name):
        """this method blocks until the product with the passed product name is produced"""
        mqtt_client = mqtt.Client()
        mqtt_client.on_message = functools.partial(mqtt_callback,
                                                mqtt_client=mqtt_client,
                                                augmented_product_name=augmented_product_name)
        mqtt_client.connect(self.get_mqtt_hostname())
        mqtt_topics = [
            (self.create_digitaltwin_subscription(cppu_name, '/States/Message'), 0)
            for cppu_name in self.get_storage_names()
        ]
        mqtt_client.subscribe(mqtt_topics)
        mqtt_client.subscribe("AI/TrainingLoop")
        mqtt_client.loop_forever()
        
    def delete_product(self, cppu_name, product):
        pass

    def reset_simulator(self):
        """reset the simulator by deleting all products present at any CPPU"""
        for product, cppu_name in self.get_products(self.all_cppu_names):
            self.delete_product(cppu_name, product)

    def get_storage_names(self):
        """return the names of the units that should have the store skill"""
        return [name for name, config in self.config['plant_layout'].items()
                if config.get('hasStore', False)]

    def delete_products_after_condition(self, augmented_product_name, cppu_threshold_detected = None):
        """return the product information as a dictionary and deletes the product from the storage"""
        self.logger.info(f"Entering Deletion of Products: {augmented_product_name}")
        with self.storage_lock:
            product_info = None
            product_type = None
            products_in_storage = self.get_products(self.get_storage_names())
            self.logger.info(f"Products from CPPU Storage: {products_in_storage}")
            for product_id, cppu_name in products_in_storage:
                product_info = self.get_product(cppu_name, product_id)
                if product_info['Configuration']['Product'] == augmented_product_name:
                    self.logger.debug(f"Production Finished: Deleting Product {product_id} from CPPU Storage {cppu_name}")
                    self.delete_product(cppu_name, product_id)
                    self.logger.info(f"Production Finished: Deleted Product {product_id} from CPPU Storage {cppu_name}")
                    product_type = augmented_product_name.split(':')[0]
                    break
                else:
                    self.logger.info(f"Production Finished: Product {product_id} in CPPU Storage {cppu_name} not corresponding to {augmented_product_name}")
            if product_type is None:
                products_not_in_storage = self.get_products(self.all_cppu_names)
                for product_id, cppu_name in products_not_in_storage:
                    if cppu_name == cppu_threshold_detected:
                        product_info = self.get_product(cppu_name, product_id)
                        self.delete_product(cppu_name, product_id)
                        self.logger.info(f"Production Threshold: Deleted Product {product_id} from CPPU {cppu_name}")
            else:
                self.logger.debug(f"Production Threshold: ELSE CONDITION {product_id} from CPPU {cppu_name}")
        return product_type, product_info
    
    def get_skill_distance_from_port(self, skill, port):
        """Given the required skill and the current port,
        compute the number of hops before reaching 
        an agent able to handle the task required"""
        port_info = self.ports_info[port]
        skills_info = port_info['States']['Skills']
        if skill in skills_info:
            plugged_equipment = skills_info[skill]['PluggedEquipment']
            distance = min([equipment_info['Distance'] for equipment_info in plugged_equipment.values()])
            return distance
        else:
            # ... otherwise return the maximum possible CPPU hops value
            return len(self.real_cppu_names) - 1

    def get_product_names(self):
        return list(self.products.keys())

    def start_produce_product(self, product_name):
        """start production, returns an augmented product name with unique uuid"""
        pass
    
    def get_counter(self, counter_key):
        if counter_key in self.counter_dict.keys():
            count_per_product = self.counter_dict[counter_key]
        else:
            count_per_product = 0
        return 1 if count_per_product > 0 else 0
    
    def return_observation_tuples(self, product_info, client_server, algorithm):
        """ Return observation based on the type of algorithm being used """
        # get observation BEFORE any production skill execution
        observation_before = self.get_previous_observation(product_info)
        self.update_counter_dict(observation_before)
        if not client_server and algorithm == 'Dist_Q':
            self.logger.info(f'{self.cppu_name}: observation before skill execution is {observation_before}')
            # get observation AFTER production skill execution
            observation_after = self.get_observation(product_info)
            self.logger.info(f'{self.cppu_name}: observation after skill execution is {observation_after}')
            observation = (observation_before, observation_after)
        elif client_server:
            observation_rllib = self.map_observation_to_rllib(observation_before)
            self.logger.info('RlLib observation_before: {}'.format(observation_rllib))
            observation = (observation_before, observation_rllib)
        return observation
    
    
    def get_observation(self, product_info):
        """ Extract information from the Simulator and construct the observation"""
        products_in_cppu = list(self.get_products(self.cppu_name))
        products_in_cppu_name = [self.get_product(self.cppu_name, product_id[0])
                                 ['Configuration']['BOP'][0]['Parameters']['Product']
                                 for product_id in products_in_cppu]
        products_encoding_per_cppu = self.get_one_hot_products_encoding(products_in_cppu_name)
        current_step = product_info['States']['CurrentBOPStep']
        current_product_name = product_info['Configuration']['Product'].split(':')[0]
        required_skill = product_info['Configuration']['BOP'][current_step]['Skill']
        skills_executed_per_product = self.get_skills_history(product_info)
        next_skills_encoding = self.get_update_skill_encoding(product_info['Configuration']['BOP']
                                                              [0]['Parameters']['Product'], skills_executed_per_product)
        previous_cppu = self.get_previous_cppu_name(product_info)

        counter_key_list = []
        if self.observation_space_dict.get('product_name', False): 
            counter_key_list.append(current_product_name)
        if self.observation_space_dict.get('next_skills', False):
            counter_key_list.append(next_skills_encoding)
        if self.observation_space_dict.get('previous_cppu', False):
            counter_key_list.append(previous_cppu)
        
        counter_key = get_string_observation(counter_key_list)
        counter = self.get_counter(counter_key)

        observation = []
        for field_name in self.observation_space_dict.keys():
            if field_name == "next_skill":
                observation.append(required_skill)
            elif field_name == "product_name":
                observation.append(current_product_name)
            elif field_name == "cppu_state":
                observation.append(products_encoding_per_cppu)
            elif field_name == "next_skills":
                observation.append(next_skills_encoding)
            elif field_name == "counter":
                observation.append(counter)
            elif field_name == "previous_cppu":
                observation.append(previous_cppu)

        observation = tuple(observation)
        if self.use_masking:
            observation = self.get_masking_observation(observation)
        return observation
    
    def get_previous_observation(self, product_info):

        # TODO: Products in the previous cppu
        products_encoding_per_cppu = [1]
        previous_skills_executed, previous_skill = self.get_previous_skills(product_info)
        current_product_name = product_info['Configuration']['Product'].split(':')[0]
        next_skills_encoding = self.get_update_skill_encoding(product_info['Configuration']['BOP']
                                                              [0]['Parameters']['Product'], previous_skills_executed)
        previous_cppu = self.get_previous_cppu_name(product_info)

        counter_key_list = []
        if self.observation_space_dict.get('product_name', False):
            # TODO: Previous product name (if assembled)
            counter_key_list.append(current_product_name)
        if self.observation_space_dict.get('next_skills', False):
            counter_key_list.append(next_skills_encoding)
        if self.observation_space_dict.get('previous_cppu', False):
            counter_key_list.append(previous_cppu)

        counter_key = get_string_observation(counter_key_list)
        counter = self.get_counter(counter_key)

        observation = []
        for field_name in self.observation_space_dict.keys():
            if field_name == "next_skill":
                observation.append(previous_skill)
            elif field_name == "product_name":
                observation.append(current_product_name)
            elif field_name == "cppu_state":
                observation.append(products_encoding_per_cppu)
            elif field_name == "next_skills":
                observation.append(next_skills_encoding)
            elif field_name == "counter":
                observation.append(counter)
            elif field_name == "previous_cppu":
                observation.append(previous_cppu)

        observation = tuple(observation)
        return observation
    
    def get_previous_skills(self, product_info):
        """Skills executed before self"""
        previous_skills_executed = []
        last_skill_by_cppu = None
        
        skill_history = product_info['States']['SkillHistory']
        i = -1
        while True:
            if abs(i) > len(skill_history):
                break
            last_entry = skill_history[i]
            if last_entry.get("Cppu") == self.cppu_name:
                i-=1
                last_skill_by_cppu = last_entry.get("Skill")
                
            else:
                for skill_info in skill_history[:i]:
                    if skill_info["Skill"] != "transport":
                        previous_skills_executed.append(skill_info["Skill"])
                break

        if last_skill_by_cppu is not None:
            # The last skill this cppu has executed is the one requested by the previous cppu
            previous_skill = last_skill_by_cppu
        else:
            # If this cppu has not done anything, the previous skill is the same as the current next skill
            current_step = product_info['States']['CurrentBOPStep']
            previous_skill = product_info['Configuration']['BOP'][current_step]['Skill']
        return previous_skills_executed, previous_skill
    
    
    def get_observation_baseline(self, product_info, next_skill = None):
        if next_skill is not None:
            required_skill = next_skill
        else:
            current_step = product_info['States']['CurrentBOPStep']
            required_skill = product_info['Configuration']['BOP'][current_step]['Skill']
        distance_array = [self.get_skill_distance_from_port(required_skill, port) for port in self.ports]
        return np.array(distance_array)
    
    
    def map_observation_to_rllib(self, observation):
        """Map the observation to a gym-compatible observation"""
        
        obs_rllib = []

        if self.observation_space_dict.get("next_skill", False):
            skill_name = observation[self.get_position_in_observation('next_skill')]
            assert isinstance(skill_name, str)
            obs_rllib.append(self.skill_names.index(skill_name))
        if self.observation_space_dict.get("product_name", False):
            product_name = observation[self.get_position_in_observation('product_name')]
            assert isinstance(product_name, str)
            obs_rllib.append(self.product_names.index(product_name))
        if self.observation_space_dict.get("cppu_state", False):
            cppu_state = observation[self.get_position_in_observation('cppu_state')]
            assert isinstance(cppu_state, list)
            obs_rllib.append(binary_list_to_int(cppu_state))
        if self.observation_space_dict.get("next_skills", False):
            next_skills = observation[self.get_position_in_observation('next_skills')]
            assert isinstance(next_skills, list)
            obs_rllib.append(binary_list_to_int(next_skills))
        if self.observation_space_dict.get("counter", False):
            counter = observation[self.get_position_in_observation('counter')]
            obs_rllib.append(counter)
        if self.observation_space_dict.get("previous_cppu", False):
            previous_cppu = observation[self.get_position_in_observation('previous_cppu')]
            assert isinstance(previous_cppu, str)
            cppu_encoding = self.cppu_dictionary.get(previous_cppu)
            obs_rllib.append(cppu_encoding)
        self.logger.info('RLlib Observation: {}'.format(obs_rllib))
        return np.array(obs_rllib)
    
    
    def get_previous_cppu_name(self, product_info):
        skill_history = product_info['States']['SkillHistory']
        i = -1
        while True:
            if abs(i) > len(skill_history):
                previous_cppu = self.cppu_name
                break
            last_entry = skill_history[i]
            if last_entry.get("Cppu") == self.cppu_name:
                i-=1
            else:
                previous_cppu = last_entry.get("Cppu")
                break
        self.logger.debug('Previous cppu was: {}'.format(previous_cppu))
        return previous_cppu
    

    def extract_kpi(self, skill_history_item):
        kpi_duration = skill_history_item["Duration"]
        idle_energy_consumption = skill_history_item['IdleEnergyConsumption']
        dynamic_energy_consumption = sum([behavior['DynamicEnergyConsumption']
                                          for behavior in skill_history_item['Behaviors']])
        kpi_energy = idle_energy_consumption + dynamic_energy_consumption
        kpi = self.fact_duration*kpi_duration + self.fact_energy*kpi_energy
        return kpi

    def get_masking_observation(self, observation):
        obs = {}
        action_mask = None
        obs["observations"] = observation
        obs["action_mask"] = action_mask
        return obs