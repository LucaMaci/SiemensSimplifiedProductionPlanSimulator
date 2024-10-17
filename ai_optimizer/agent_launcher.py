import subprocess
import os
import signal
import json
import time
import argparse
import socket
import numpy as np
import paho.mqtt.client as mqtt
import threading
import sys
sys.path.insert(0, '..')
from simulator_stub import Simulator
from logging_formatter import setup_logger

ENV_OPTIMIZER = 'CPPS_OPTIMIZER_HOME'
CONFIG_PATH = "../config/simulator_config.json"
SIMULATOR_CONFIG_PATH = "configs/simulator_config_3units.json"


# TODO: Extract everything MQTT related in a separate file to avoid redefinition (e.g. like these vars are redefined in server_creator.py)
# MQTT topics
#   all topics are placed in the AI namespace
SERVER_BASE_TOPIC = 'AI'
SERVER_READY_TOPIC = 'ServerReady'

simulator = None
logger = setup_logger(f'AgentLauncher::{os.getpid()}')
launcher_processes = dict()
launcher_servers = dict()

def wait_for_servers(cppu_ready):
    logger.info("Waiting for servers...")
    while len(cppu_ready) < len(simulator.real_cppu_names):
        time.sleep(0.1)
    logger.info("All the servers are READY!")
    cppu_ready = []

def find_port(port=8000):
    """Find a port not in ues starting at given port"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        if s.connect_ex(("localhost", port)) == 0:
            return find_port(port=port + 1)
        else:
            return port
        
def get_available_ports(n, baseport):
    available_ports = [find_port(port=baseport)]
    while len(available_ports) < n:
        available_ports.append(find_port(port=available_ports[-1]+1))
    return available_ports

def getvar(varname, val, default=''):
    if val is not None:
        os.environ[varname] = val
    elif varname not in os.environ:
        os.environ[varname] = default
    return os.environ[varname]

def terminate_servers():
    logger.info("Stopping servers:")
    try:
        for cppu, process in launcher_servers.items():
            logger.info(f"  > Killing {cppu} server ... ")
            process.kill()
    except Exception as e:
        logger.error(f"Fatal error while stopping the servers. Quitting.", e)

def terminate():
    try:
        timeout = False
        # try to terminate all these processes with SIGTERM ...
        logger.info(f"Stopping optimizer:")
        for cppu, process in launcher_processes.items():
            logger.info(f"  > Stopping processes for {cppu} ... ")
            process.terminate()
            try:
                process.wait(timeout=5)
            except:
                timeout = True

        if timeout:
            logger.warning(f"(some) Processes refuse to stop gracefully. Killing:")
            # after a timeout, kill all processes with SIGKILL
            for cppu, process in launcher_processes.items():
                logger.info(f"  > Killing {cppu} processes ... ")
                process.kill()
                process.wait()

        if simulator:
            logger.info(f"Stopping the simulator")
            simulator.terminate()
        else:
            logger.warning(f"Not stopping the simulator (was not started)")

    except Exception as e:
        logger.error(f"Fatal error while stopping the optimizer. Quitting.", e)

    if client_server:
        terminate_servers()
        
    os.system('sudo killall -9 digitaltwin timekeeper skillcontrol timeclient equipmentcontrol')
    logger.info(f"Done. Bye!")
    exit()


def get_cli_args():
    """Create CLI parser and return parsed arguments"""
    parser = argparse.ArgumentParser()

    # Example-specific args.
    parser.add_argument(
        '--learning_config_path',
        type=str,
        help="The path of the learning configuration"
    )

    parser.add_argument(
        '--out_dir',
        type=str,
        default="output/",
        help="The output directory"
    )

    parser.add_argument(
        '--timeout',
        type=int,
        help="Timeout for single experiment"
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=43,
        help="Seed base for all the experiments"
    )

    return parser.parse_args()

def setup_mqtt_client():
    mqtt_hostname = simulator.get_mqtt_hostname()

    mqtt_sub_client.connect(mqtt_hostname)
    mqtt_sub_client.on_message = mqtt_sub_callback
    mqtt_sub_client.subscribe('/'.join([SERVER_BASE_TOPIC, SERVER_READY_TOPIC, '#']))
    mqtt_sub_client.loop_start()
    logger.info('MQTT subscriber started')

def add_cppus(payload, cppu_name):
    with cppu_ready_lock:
        cppu_ready.append(cppu_name)
        logger.info('Appended {} to cppu_ready: {}'. \
                        format(cppu_name, cppu_ready))

def mqtt_sub_callback(client, userdata, message):
    parameters = message.topic.split('/')
    payload = message.payload.decode('utf-8')
    logger.info(f'Received message {message.topic} with payload {payload}')
    if (parameters[0] == SERVER_BASE_TOPIC
        and parameters[1] == SERVER_READY_TOPIC):
        payload = json.loads(payload)
        threading.Thread(target=add_cppus,
                        args=(int(payload), parameters[1])).start()

if __name__ == "__main__":

    args = get_cli_args()
    np.random.seed(args.seed)
    max_seed = 100000

    # add signal hooks
    signal.signal(signal.SIGINT, terminate)
    signal.signal(signal.SIGQUIT, terminate)
    signal.signal(signal.SIGTERM, terminate)

    logger.info(f'Starting the CPPS simulator')

    with open(CONFIG_PATH) as config_file:
        config = json.load(config_file)
    OUTPUT_PATH = config['output_path']
    TRAJECTORY_PATH = config['trajectory_path']
    simulator = Simulator(SIMULATOR_CONFIG_PATH)

    logger.info(f"Starting the CPPS optimizer")
    path_here = os.path.realpath(os.path.dirname(__file__))

    out_dir = './output' if args.out_dir is None else args.out_dir
    learning_config_path = getvar(ENV_OPTIMIZER, args.learning_config_path, os.path.join(path_here, 'configs',
                                                                                         'learning_config.json'))

    agent_creator_path = os.path.join(path_here, 'cppu_path_learner_decentralized.py')
    server_creator_path = os.path.join(path_here, 'server_creator.py')


    with open(learning_config_path) as config_file:
        learning_config = json.load(config_file)

    client_server = learning_config['client_server']
    algorithm = learning_config["algorithm_class"]

    try:

        # Launch Policy Servers
        if client_server:

            SERVER_ADDRESS = learning_config["SERVER_ADDRESS"]
            SERVER_BASE_PORT = learning_config["SERVER_BASE_PORT"]
            num_workers = learning_config["number_workers"]
            restore_checkpoint = learning_config["restore_checkpoint"]
            
            available_ports = get_available_ports(n=len(simulator.real_cppu_names),
                                                  baseport=SERVER_BASE_PORT)
            logger.debug('AVAILABLE PORTS: {}'.format(available_ports))

            cppu_ready = []
            cppu_ready_lock = threading.Lock()
            
            mqtt_sub_client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION1)
            setup_mqtt_client()
            
            for idx, cppu_name in enumerate(simulator.real_cppu_names):

                cppu_dir = os.path.join(out_dir, 'rllib_server_log', cppu_name)
                os.makedirs(cppu_dir, exist_ok=True)

                LOCAL_PORT = available_ports[idx]

                seed = np.random.randint(max_seed)
                logger.debug(f'Policy_server {cppu_name}: seed is {seed}')

                cmd = ['python3', server_creator_path,
                        '--addr', str(SERVER_ADDRESS),
                        '--port', str(LOCAL_PORT),
                        '--scp', str(simulator.path_config),
                        '--product_config_path', simulator.path_products_config,
                        '--lcp', str(learning_config_path),
                        '--workers', str(num_workers),
                        '--cppu', str(cppu_name),
                        '--algo', str(algorithm),
                        '--checkpoint', str(restore_checkpoint),
                        '--outdir', out_dir,
                        '--seed', str(seed)]
                logger.debug(f'Launching Policy Server::{SERVER_ADDRESS}:{LOCAL_PORT}')
                logger.debug(f" > {' '.join(cmd)}")
                f_out = os.path.join(cppu_dir, "server_{}_out.log".format(cppu_name))
                f_err = os.path.join(cppu_dir, "server_{}_err.log".format(cppu_name))
                print(' '.join(cmd[2:]))
                #launcher_servers[cppu_name] = subprocess.Popen(cmd, stdout=open(f_out, "w"), stderr=open(f_err, "w"))
            # wait_for_servers(cppu_ready)

        # Launch agents
        for idx, cppu_name in enumerate(simulator.real_cppu_names):

            # # --------------------------------------------------------------------------------------------------------

            seed = np.random.randint(max_seed)
            logger.debug(f'agent_launcher {cppu_name}: seed is {seed}')

            cmd = ['/usr/bin/python3', agent_creator_path,
                   '--cppu_name', cppu_name,
                   '--learning_config_path', learning_config_path,
                   '--simulator_config_path', simulator.path_config,
                   '--product_config_path', simulator.path_products_config,
                   '--seed', str(seed),
                   '--out_dir', out_dir]
            
            if client_server:
                cmd += ['--serverport', str(available_ports[idx])]

            logger.info(f'Launching Agent::{cppu_name}')
            logger.debug(f" > {' '.join(cmd)}")
            print(' '.join(cmd[2:]))
            #launcher_processes[cppu_name] = subprocess.Popen(cmd)

        # Launch orchestrator
        # cppu_name = 'cppc'
        # cmd = ['/usr/bin/python3', agent_creator_path,
        #        '--orchestrator', str(True),
        #        '--cppu_name', cppu_name,
        #        '--learning_config_path', learning_config_path,
        #        '--simulator_config_path', simulator.path_config,
        #        '--product_config_path', simulator.path_products_config,
        #        '--out_dir', out_dir,
        #        '--seed', str(args.seed)]  # save the original seed of the experiment
        # logger.info(f'Launching Agent::Orchestrator')
        # logger.debug(f" > {' '.join(cmd)}")
        # print(' '.join(cmd[2:]))
        #launcher_processes[cppu_name] = subprocess.Popen(cmd)

    except Exception as e:
        logger.error(f'Error in launching for Agent {cppu_name}: {str(e)}')
        terminate()

    logger.info("Launching of optimizer complete. Waiting for the simulation to finish ...")

    logger.info('Waiting for Orchestrator to terminate...')
    # try:
    #     launcher_processes['cppc'].wait(args.timeout)
    #     logger.info('Orchestrator: execution completed!')
    #     terminate()
    #     time.sleep(30)  # give some time for cleaning up simulator ports
    # except Exception as e:
    #     logger.error('TIMEOUT!')
    #     terminate()
    #     time.sleep(30)
    #     raise e

