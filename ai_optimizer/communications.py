import aiohttp
import asyncio
from uuid import uuid4
from traceback import TracebackException
from types import TracebackType
from typing import List


class Communications:
    """
    Class that will handle the comms between ai-optimizer and skill framework
    """

    def __init__(self, rest_gateway: str, cppu_name: str = None) -> None:
        self.rest_gateway = rest_gateway
        self.cppu_name = cppu_name
        self._session = aiohttp.ClientSession()

    async def __aenter__(self) -> "Communications":
        """Context entry descriptor that returns itself for an async session

        Returns:
            Communications: itself.
        """
        return self

    async def __aexit__(
        self,
        exc_type: Exception,
        exc_val: TracebackException,
        traceback: TracebackType,
    ) -> None:
        """Context exit descriptor that closes the coroutine gracefully

        Args:
            exc_type (Exception): General type exception
            exc_val (TracebackException): Exception on the coroutine
            traceback (TracebackType): Traceback for the coroutine exception
        """
        await self.close()

    async def close(self) -> None:
        """Sesssion explict close clause"""
        await self._session.close()

    async def get_health(self, cppu_name: str = None) -> int:
        """GET call to prove the health of the digital twin endpoint

        Args:
            cppu_name (str, optional): Name of the cppu to test. Defaults to None.

        Returns:
            int: Status code from the call.
        """
        if cppu_name is None:
            cppu_name = self.cppu_name
        url = f"{self.rest_gateway}/{cppu_name}/digitaltwin/health/ready"
        async with self._session.get(url) as response:
            response.raise_for_status()
            return response.status

    async def get_digitaltwin_state(self, cppu_name: str = None) -> bool:
        """GET call, digital twin state.

        Args:
            cppu_name (str, optional): CPPU name string. Defaults to None.

        Returns:
            bool: True if ready or running.
        """
        if cppu_name is None:
            cppu_name = self.cppu_name
        url = f"{self.rest_gateway}/{cppu_name}/digitaltwin/states"
        async with self._session.get(url) as response:
            response.raise_for_status()
            status = await response.json()
            if status["Ready"] is True or status["Running"] is True:
                return True
            return False

    async def get_equipmentcontrol_state(self, cppu_name: str = None) -> dict:
        """GET call for the equipment control

        Args:
            cppu_name (str, optional): CPPU string name. Defaults to None.

        Returns:
            dict: JSON with the current state of the CPPU. 
        """
        if cppu_name is None:
            cppu_name = self.cppu_name
        url = f"{self.rest_gateway}/{cppu_name}/equipmentcontrol/getstate"
        async with self._session.get(url) as response:
            response.raise_for_status()
            status = await response.json()
            return status

    async def get_skills(self, cppu_name: str = None) -> dict:
        """GET call for the skills available in the unit

        Args:
            cppu_name (str, optional): String with the assigned unit name.
            Defaults to None.

        Returns:
            dict: Dict with all the skills possible in the unit.
        """
        if cppu_name is None:
            cppu_name = self.cppu_name
        url = f"{self.rest_gateway}/{cppu_name}/digitaltwin/skills"
        async with self._session.get(url) as response:
            response.raise_for_status()
            return await response.json()

    async def get_ports(self, cppu_name: str = None) -> dict:
        """GET call to list the ports in an unit

        Args:
            cppu_name (str, optional): String with the assigned unit name.
            Defaults to None.

        Returns:
            dict: dict with all of the ports possible in the unit.
        """
        if cppu_name is None:
            cppu_name = self.cppu_name
        url = f"{self.rest_gateway}/{cppu_name}/digitaltwin/ports"
        async with self._session.get(url) as response:
            response.raise_for_status()
            return await response.json()

    async def get_ports_deep(self, cppu_name: str = None) -> dict:
        """GET call to list the ports in an unit in detail

        Args:
            cppu_name (str, optional): String with the assigned unit name.
            Defaults to None.

        Returns:
            dict: dict with all the possible ports in detail of the unit.
        """
        if cppu_name is None:
            cppu_name = self.cppu_name
        url = f"{self.rest_gateway}/{cppu_name}/digitaltwin/ports/deep"
        async with self._session.get(url) as response:
            response.raise_for_status()
            return await response.json()

    async def get_port(self, port: int, cppu_name: str = None) -> dict:
        """GET call to show the information of a port in an unit

        Args:
            port (int): Int of a port in an unit
            cppu_name (str, optional): String with the assigned unit name.
            Defaults to None.

        Returns:
            dict: dict with all the single port data available.
        """
        if cppu_name is None:
            cppu_name = self.cppu_name
        url = f"{self.rest_gateway}/{cppu_name}/digitaltwin/port/{port}"
        async with self._session.get(url) as response:
            response.raise_for_status()
            return await response.json()

    async def get_product(self, product: str, cppu_name: str = None) -> dict:
        """GET call to get the information of a product in the line

        Args:
            product (str): String of the product ID
            cppu_name (str, optional): String with the assigned unit name.
            Defaults to None.

        Returns:
            dict: dict with a product information in a unit.
        """
        if cppu_name is None:
            cppu_name = self.cppu_name
        url = f"{self.rest_gateway}/{cppu_name}/digitaltwin/product/{product}"
        async with self._session.get(url) as response:
            response.raise_for_status()
            return await response.json()

    async def get_products(self, cppu_name: str = None) -> dict:
        """GET call to get all the products available in an unit

        Args:
            cppu_name (str, optional): String with the assigned unit name.
            Defaults to None.

        Returns:
            dict: dict with all the products in an unit.
        """
        if cppu_name is None:
            cppu_name = self.cppu_name
        url = f"{self.rest_gateway}/{cppu_name}/digitaltwin/products"
        async with self._session.get(url) as response:
            response.raise_for_status()
            return await response.json()

    async def get_whiteboard(self, cppu_name: str = None) -> dict:
        """GET call to get the whiteboard state of an unit.

        Args:
            cppu_name (str, optional): String with the assigned unit name.
            Defaults to None.

        Raises:
            ValueError: in case there is no whiteboard in the response

        Returns:
            dict: dict with the current whiteboard state of an unit.
        """
        if cppu_name is None:
            cppu_name = self.cppu_name
        url = f"{self.rest_gateway}/{cppu_name}/digitaltwin/state/Whiteboard"
        async with self._session.get(url) as response:
            response.raise_for_status()
            state = await response.json()
            if "Whiteboard" not in state:
                raise ValueError("There's no whiteboard state in the digital twin")
            return state["Whiteboard"]

    async def post_whiteboard(self, payload: dict, cppu_name: str = None) -> dict:
        """POST call to set a state on an specific unit

        Args:
            payload (dict): dictionary with the desired state
            cppu_name (str, optional): String with the assigned unit name.
            Defaults to None.

        Returns:
            dict: confirmation of the payload.
        """
        if cppu_name is None:
            cppu_name = self.cppu_name
        if "Whiteboard" not in payload:
            payload = {"Whiteboard": payload}
        url = f"{self.rest_gateway}/{cppu_name}/digitaltwin/state/Whiteboard"
        async with self._session.post(url=url, json=payload) as response:
            response.raise_for_status()
            return response

    async def post_produce_product(
        self, product_name: str, product_data: dict, cppu_name: str = None
    ) -> dict:
        """POST call to produce a product in a line.

        Args:
            product_name (str): product string as defined in the config file
            product_data (dict): product description with all the BOP and general info.
            cppu_name (str, optional): String with the assigned unit name.
            Defaults to None.

        Returns:
            dict: a dict with an UIID number that identifies the product.
        """
        if cppu_name is None:
            cppu_name = self.cppu_name
        url = f"{self.rest_gateway}/{cppu_name}/skillcontrol/produce/start"
        augumented_product_name = f"{product_name}:{uuid4()}"
        payload = {"Context": augumented_product_name, "Payload": product_data}
        async with self._session.post(url=url, json=payload) as response:
            response.raise_for_status()
            return augumented_product_name

    async def delete_product(self, product: str, cppu_name: str = None) -> int:
        """DELETE call to eliminate a product from the line

        Args:
            product (str): Unique ID from the product.
            cppu_name (str, optional): CPPU identifier. Defaults to None.

        Returns:
            int: Status response code
        """
        url = f"{self.rest_gateway}/{cppu_name}/digitaltwin/product/{product}"
        if cppu_name is None:
            cppu_name = self.cppu_name
        async with self._session.delete(url) as response:
            response.raise_for_status()
            return await response.status

    async def get_observation(self):
        pass


async def get_supply_units(cppu_names: List[str], class_args: dict) -> List[str]:
    """Get all units with the skill supply

    Args:
        cppu_names (List[str]): send the factory layout
        class_args (dict): send the communication class init dict.

    Returns:
        List[str]: list of all the supply units
    """
    async with Communications(**class_args) as session:
        tasks = list()
        for cppu in cppu_names:
            tasks.append(session.get_skills(cppu))
        result = await asyncio.gather(*tasks)
    supply_units = list()
    for index, skills in enumerate(result):
        if "supply" in skills:
            supply_units.append(cppu_names[index])
    return supply_units


async def get_store_units(cppu_names: List[str], class_args: dict) -> List[str]:
    """Get all units with the store skill

    Args:
        cppu_names (List[str]): send the factory layout
        class_args (dict): send the communication class init dict.

    Returns:
        List[str]: list of all the store units
    """
    async with Communications(**class_args) as session:
        tasks = list()
        for cppu in cppu_names:
            tasks.append(session.get_skills(cppu))
        result = await asyncio.gather(*tasks)
    store_units = list()
    for index, skills in enumerate(result):
        if "store" in skills:
            store_units.append(cppu_names[index])
    return store_units


async def delete_all_products(cppu_names: List[str], class_args: dict) -> List:
    """Delete all the products from all the units

    Args:
        cppu_names (List[str]): send the factory layout
        class_args (dict): send the communication class init dict.

    Returns:
        List: list with the query result.
    """
    async with Communications(**class_args) as session:
        tasks = list()
        for cppu in cppu_names:
            tasks.append(session.get_products(cppu))
        result = await asyncio.gather(*tasks)
        tasks = list()
        for index, products in enumerate(result):
            if products:
                continue
            for product in products:
                tasks.append(session.delete_product(product, cppu_names[index]))
        result = await asyncio.gather(*tasks)
        return result


if __name__ == "__main__":
    units = [f"cppu{number}" for number in range(0, 10)]
    units.remove("cppu3")
    units.remove("cppu7")
    class_args = {"rest_gateway": "http://127.0.0.1", "cppu_name": "line0"}
    print(asyncio.run(get_supply_units(units, class_args)))
