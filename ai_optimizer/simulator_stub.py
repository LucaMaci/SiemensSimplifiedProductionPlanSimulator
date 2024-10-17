import json


class Simulator():
    def __init__(self, confing_path, mqtt_hostname="localhost"):
        self.config_path = self.path_config = self.path_products_config = confing_path
        with open(confing_path) as config_file:
            config = json.load(config_file)
        self.n_agents = config["n_agents"]
        self.mqtt_hostname = mqtt_hostname
        self.real_cppu_names = [f'cppu_{i}' for i in range(self.n_agents)]

    def terminate(self):
        pass

    def get_mqtt_hostname(self):
        return self.mqtt_hostname