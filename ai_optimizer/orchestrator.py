import json
from os import path
from random import choices
from typing import List, Dict, Tuple, Generator
from src.logging_formatter import LoggingFormatter


class Skill:
    def __init__(self) -> None:
        pass


class Product:
    def __init__(self) -> None:
        pass


class Agent:
    def __init__(self) -> None:
        pass

    def _read_config(self) -> None:
        pass


class Orchestrator:
    """
    Orchestrator class containing the layout, product and agent deployment for
    trainig and inference on RL systems.
    """

    def __init__(
        self,
        product_config_dir: str = None,
        logger: LoggingFormatter = None,
        layout_config: str = None,
        learning_config: str = None,
        mqtt_host_url: str = "tcp://127.0.0.1:1883",
        gateway_url: str = "http://127.0.0.1",
    ) -> None:
        self.logger = logger
        self._product_config_dir = product_config_dir
        self._product_dict = self._read_product_config()
        self._layout_config_dir = layout_config
        self._layout_dict = self._read_layout_dict()
        self._learning_config_dir = learning_config
        self.mqtt_host_url = mqtt_host_url
        self.gateway_url = gateway_url

    @property
    def layout_config_dir(self) -> str:
        """location for the plant layout config file

        Raises:
            FileExistsError: Config file does not  exist

        Returns:
            str: a verified layout config file
        """
        if self._layout_config_dir is None:
            return None
        if not path.exists(self._layout_config_dir):
            raise FileExistsError("Layout config file does not exist")
        return self._layout_config_dir

    @property
    def all_cppu_names(self) -> List[str]:
        """property that defines all the names of the CPPU's in the plant

        Returns:
            List[str]: a list of string with the defined names
        """
        return [name for name in self._layout_dict]

    @property
    def real_cppu_names(self) -> List[str]:
        """property that defines all the CPPU's that are not the plant

        Returns:
            List[str]: a list of strings with the cppu names
        """
        return [
            name
            for name, config in self._layout_dict.items()
            if config["kind"] == "Unit"
        ]

    @property
    def cppc_names(self) -> List[str]:
        """property that defines the names of the plant

        Returns:
            List[str]: a list of strings with plant names
        """
        return [
            name
            for name, config in self._layout_dict.items()
            if config["kind"] in ("Plant", "Line")
        ][0]

    @property
    def storage_names(self) -> List[str]:
        """property that defines the names of the CPPU's with storage capabilities

        Returns:
            List[str]: a list of strings with cppu's with storage
        """
        return [
            name
            for name, config in self._layout_dict.items()
            if config.get("hasStore", False)
        ]

    @property
    def product_config_dir(self) -> str:
        """Property that defines a product configuration

        Raises:
            FileExistsError: In case the config file does not exist

        Returns:
            str: a verified config file path
        """
        if self._product_config_dir is None:
            return None
        if not path.exists(self._product_config_dir):
            raise FileExistsError("Product config file does not exist")
        return self._product_config_dir

    @property
    def product_names(self) -> List[str]:
        """Read-only property defined by the config file with the product names

        Returns:
            List[str]: List of product names
        """
        return self._read_product_components(
            list(self._product_dict.keys()), self._product_dict
        )

    @property
    def product_type_number(self) -> int:
        """Read-only property defined by the config file with
           the number of distinct products

        Returns:
            int: Int with the number of types of products
        """
        return len(self.product_names)

    @property
    def product_quantities(self) -> List[Tuple[str, int]]:
        """Read-only property defined by the config file with a list of quantities of
           distinct products

        Returns:
            List[Tuple[str, int]]: list of quantities per product
        """
        return self._read_product_quantities(
            list(self._product_dict.keys()), self._product_dict
        )

    @property
    def skill_names(self) -> List[str]:
        """Read-only property defined by the config file with a list of skill names
           defined on the BOP of the products

        Returns:
            List[str]: list of skills for each product
        """
        return self._read_skills_for_products(
            list(self._product_dict.keys()), self._product_dict
        )

    @property
    def skill_product_pair(self) -> List[str]:
        """Read-only property, a pair of skills and products.

        Returns:
            List[str]: List of the products and skills.
        """
        return self._get_product_skills_pairs(
            list(self._product_dict.keys()), self._product_dict
        )

    @property
    def learninig_config_dir(self) -> str:
        """Read-only property, the directory for the learning config.

        Raises:
            FileExistsError: Raises an error when the file does not exist.

        Returns:
            str: A string containing a valid directory.
        """
        if self._learning_config_dir is None:
            return None
        if not path.exists(self._learning_config_dir):
            raise FileExistsError("Learning config file does not exist")
        return self._learning_config_dir

    @property
    def total_product_number_to_produce(self) -> int:
        """Read only property defined as the amount of products to produce.

        Returns:
            int: Total number of products to produce in a run
        """
        total = 0
        for product in self.product_quantities:
            total = total + product[1]
        return total

    @property
    def total_number_to_produce(self) -> List[str]:
        """Read-only property list of every instance of a product to produce taking 
        into account the quantities.

        Returns:
            List[str]: List of product to produce.
        """
        total_number_to_produce = list()
        for products in self.product_quantities:
            total_number_to_produce = (
                total_number_to_produce + [products[0]] * products[1]
            )
        return total_number_to_produce

    def _read_product_config(self) -> dict:
        """Read all of the configurations for the current product config

        Returns:
            dict: Numbers and lists of products characteristics
        """
        if self.product_config_dir is None:
            return None
        with open(self.product_config_dir) as file:
            products = json.load(file)
        return products

    def _read_layout_dict(self) -> dict:
        """Read the config file with the plant layout

        Raises:
            ValueError: There's no plant layout defined in the config file

        Returns:
            dict: a dict with the plant layout tree
        """
        if self.layout_config_dir is None:
            return None
        with open(self.layout_config_dir) as file:
            config = json.load(file)
        if "plant_layout" not in config:
            raise ValueError("Config file does not contain a plant layout")
        return config["plant_layout"]

    def _read_product_components(
        self, main_products: List[str], product_dict: Dict
    ) -> List[str]:
        """Function to read all of the product names and
           components from the config file.

        Args:
            main_products (List[str]): A list of the names of the products to include.
            product_dict (Dict): A dictionary representation of the JSON config file.

        Returns:
            List[str]: A list of all products and component names.
        """
        products_in_components = set()
        for product_name in main_products:
            products_in_components.add(product_name)
            components = product_dict[product_name]["Components"].keys()
            sub_products = self._read_product_components(
                components, product_dict[product_name]["Components"]
            )

            for sub_product in sub_products:
                products_in_components.add(sub_product)

        return list(products_in_components)

    def _read_product_quantities(
        self, main_products: List[str], product_dict: Dict
    ) -> List[Tuple]:
        """Reads the units to produce from each product

        Args:
            main_products (List[str]): List of the final product types
            product_dict (Dict): Dict with all the config file for products

        Returns:
            List[Tuple]: List of pairs of (product names and quantities)
        """
        product_quantities = list()
        for product_name in main_products:
            if "Quantity" in product_dict[product_name]["Configuration"]:
                quantity = int(product_dict[product_name]["Configuration"]["Quantity"])
            else:
                self.logger.warning(
                    f"The product {product_name}, has no defined quantity, default is 1"
                )
                quantity = 1
            product_quantity = tuple((product_name, quantity))
            product_quantities.append(product_quantity)

        return product_quantities

    def _read_skills_for_products(
        self, main_products: List[str], product_dict: Dict
    ) -> List[str]:
        """Skill names to load from the BOP

        Args:
            main_products (List[str]): List of the products
            product_dict (Dict): Complete dict of the config

        Raises:
            ValueError: In case there is no BOP defined in the config file

        Returns:
            List[str]: a list of all of the skills necessary in the script
        """
        skill_set = set()
        for product_name in main_products:
            if "BOP" in product_dict[product_name]["Configuration"]:
                bop = product_dict[product_name]["Configuration"]["BOP"]
            else:
                raise ValueError("Bill of processes is undefined")
            for skills in bop:
                skill_set.add(skills["Skill"])

        return list(skill_set)

    def _get_product_skills_pairs(
        self, main_products: List[str], product_dict: Dict
    ) -> List[str]:
        """Get product, quantity and skills

        Args:
            main_products (List[str]): List of the products to produce
            product_dict (Dict): Dict of the product config

        Returns:
            List[str]: List of tuples with product, quantity and skills
        """

        product_skills_pair = set()

        for product in main_products:
            if "Quantity" in product_dict[product]["Configuration"]:
                quantity = product_dict[product]["Configuration"]["Quantity"]
            else:
                quantity = 1
            skills_list = product_dict[product]["Configuration"]["BOP"]
            skills_to_add = []
            for skill_list in skills_list:
                skill = skill_list["Skill"]
                skills_to_add.append(skill)

            product_skills_pair.add((product, quantity, tuple(skills_to_add)))

            subproducts_dict = product_dict[product]["Components"]
            subproducts_name = product_dict[product]["Components"].keys()

            retrieved_pairs = self._get_product_skills_pairs(
                subproducts_name, subproducts_dict
            )

            for retrieved_pair in retrieved_pairs:
                product_skills_pair.add(retrieved_pair)

        return product_skills_pair

    def _read_learning_hyperparameters(self) -> dict:
        """Hyperparameter dictionary from the learining config file

        Returns:
            dict: dict with all learning hyperparameters
        """
        if self.learning_config_dir is None:
            return None
        with open(self.learninig_config_dir) as file:
            hyperparameters = json.load(file)
        return hyperparameters

    def product_probability_input(
        self, probabilities: List[float] = None
    ) -> Generator[str, str, str]:
        """Generator of a product list probabilisticaly

        Args:
            probabilities (List[float], optional): A list of probabilities for each
            product, must contain a number for each product. Defaults to None.

        Raises:
            ValueError: If the length of the list of probabilities is not the same
            as the number of products
            SyntaxError: If the probababilities do not add to 1

        Yields:
            Generator[str, str, str]: A generator of a list of product names
        """
        if probabilities is None:
            probabilities = list()
            for product_name, product_quantity in self.product_quantities:
                probabilities.append(
                    product_quantity / self.total_product_number_to_produce
                )
        if len(probabilities) != len(self.product_quantities):
            raise ValueError(
                f"Not the same number of products {len(self.product_quantities)}"
            )
        if sum(probabilities) != 1:
            raise SyntaxError(f"The probabilites do not add to 1: {sum(probabilities)}")
        production_list = choices(
            self.product_names, probabilities, k=self.total_product_number_to_produce
        )
        for product in production_list:
            yield product

    def storage_probability_input(
        self, probabilities: List[float] = None
    ) -> Generator[str, str, str]:
        """Generator of a storage list probabilistically

        Args:
            probabilities (List[float], optional): A list of probabilities for each
            storage unit, must contain a number for each product. Defaults to None.

        Raises:
            ValueError: If the length of the list of probabilities is not the same
            as the number of storage units
            SyntaxError: If the probababilities do not add to 1

        Yields:
            Generator[str, str, str]: A generator of a list of storage unit names
        """
        if probabilities is None:
            storage_units_number = len(self.storage_names)
            probabilities = [1 / storage_units_number] * storage_units_number
        if len(probabilities) != len(self.storage_names):
            raise ValueError(
                f"Not the same number of storage units {len(self.storage_names)}"
            )
        if sum(probabilities) != 1:
            raise SyntaxError(f"The probabilites do not add to 1: {sum(probabilities)}")
        storage_list = choices(
            self.storage_names, probabilities, k=len(self.storage_names)
        )
        for storage_unit in storage_list:
            yield storage_unit

    def storage_product_pair(
        self,
        product_probability: List[float] = None,
        storage_probability: List[float] = None,
    ) -> List[Tuple[str, str]]:
        """Creates a list of product storage pairs randomly

        Args:
            product_probability (List[float], optional): Optional list of product
            probabilites. Defaults to None.
            storage_probability (List[float], optional): Optional list of storage
            probabilities. Defaults to None.

        Returns:
            List[Tuple[str, str]]: A list of tuples with a product/storage pair.
        """
        product_names = list()
        storage_unit_names = list()
        for product in self.product_probability_input(product_probability):
            product_names.append(product)

        for storage_unit in self.storage_probability_input(storage_probability):
            storage_unit_names.append(storage_unit)

        return list(zip(product_names, storage_unit_names))

    def storage_product_roundrobin(self) -> List[Tuple[str, str]]:
        """Simple roundrobin pairs for product input

        Returns:
            List[Tuple[str, str]]: list of a product, storage tuple
        """
        storage_product_pairs = list()
        count = 0
        for product in self.total_number_to_produce:
            storage_product_pairs.append((product, self.storage_names[count]))
            count += 1
            if count >= len(self.storage_names):
                count = 0
        return storage_product_pairs

    def compute_reward_rllib(self, skill_history):
        """Compute the reward in a Semi-MDP fashion for RLlib
        TODO Evaluate and test the function
        """
        self.logger.debug(f"RLLib Computing KPI for skill history: {skill_history}")

        history_counter = 1
        production_kpi = 0
        overall_kpi = 0
        while history_counter <= len(skill_history):
            if skill_history[-history_counter]["Cppu"] != self.cppu_name:
                overall_kpi += self.extract_kpi(skill_history[-history_counter])
                if (
                    skill_history[-history_counter]["Skill"]
                    not in self.non_production_skill
                ):
                    production_kpi += self.extract_kpi(skill_history[-history_counter])
            else:
                break
            history_counter += 1
        self.logger.info(f"RLLib Computed KPI: {overall_kpi}")
        reshaped_reward = -self.shape_reward(
            overall_kpi=overall_kpi, production_kpi=production_kpi
        )
        self.logger.info(f"RLLib Reshaped KPI: {reshaped_reward}")
        return reshaped_reward

    def layout(self) -> None:
        """TODO evaluate layouts"""
        pass

    def compute_reward(self, skill_history):
        """Compute reward as the negative sum of the duration of the last behavior
        executed by the agent (skills + transport)
        TODO evaluate the reward function
        """
        history_counter = 1
        production_kpi = 0
        overall_kpi = 0
        while history_counter <= len(skill_history):
            if skill_history[-history_counter]["Cppu"] == self.cppu_name:
                overall_kpi += self.extract_kpi(skill_history[-history_counter])
                if (
                    skill_history[-history_counter]["Skill"]
                    not in self.non_production_skill
                ):
                    production_kpi += self.extract_kpi(skill_history[-history_counter])
            else:
                break
            history_counter += 1
        self.logger.info(f"KPI base reward: {overall_kpi}")
        reshaped_reward = -self.shape_reward(
            overall_kpi=overall_kpi, production_kpi=production_kpi
        )
        self.logger.info(f"KPI reshaped reward: {reshaped_reward}")
        return reshaped_reward
