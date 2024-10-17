import ray
from ray.rllib.env.policy_server_input import PolicyServerInput
from ray.rllib.algorithms.callbacks import MemoryTrackingCallbacks
from ray import air, tune
from ray.tune.logger import pretty_print
from ray.tune.registry import get_trainable_cls
from logging_formatter import setup_logger
import os
import argparse
import json
from agent_simulator import AgentSImulator
from custom_models import ActionMaskPolicyModel, ActionMaskSACQModel
import paho.mqtt.client as mqtt
import threading

SERVER_BASE_TOPIC = 'AI'
SERVER_READY_TOPIC = 'ServerReady'

def publish_server_ready(logger_s, simulator, name):
    mqtt_pub_client = mqtt.Client()
    mqtt_pub_client_lock = threading.Lock()
    mqtt_pub_client.connect(simulator.get_mqtt_hostname())
    mqtt_pub_client.loop_start()
    mqtt_topic = '/'.join([SERVER_BASE_TOPIC, SERVER_READY_TOPIC, name])
    payload = '1'
    retainFlag = True
    logger_s.info(f'MQTT: sending {mqtt_topic} payload: {payload} ...') 
    with mqtt_pub_client_lock:
        mqtt_pub_client.publish(mqtt_topic, payload = payload, retain  = retainFlag)
        logger_s.info(f'Message {mqtt_topic} payload: {payload} SENT with retainFlag {retainFlag}')

def get_cli_args():
    """Create CLI parser and return parsed arguments"""
    parser = argparse.ArgumentParser()

    # Example-specific args.
    parser.add_argument(
        "--addr",
        type=str,
        # default=SERVER_ADDRESS,
        help="The server address to use"  # " f"Default is {SERVER_BASE_PORT}.",
    )
    parser.add_argument(
        "--port",
        type=str,
        # default=LOCAL_PORT,
        help="The base-port to use. "  # f"Default is {SERVER_BASE_PORT}.",
    )
    parser.add_argument(
        "--scp",
        type=str,
        # default=LOCAL_PORT,
        help="Simulator config path",
    )
    parser.add_argument(
        '--product_config_path',
        type=str,
        # default=LOCAL_PORT,
        help="Product config path",
    )
    parser.add_argument(
        '--lcp',
        type=str,
        help="The path of the learning configuration"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="The number of workers"
    )
    parser.add_argument(
        "--cppu",
        type=str,
        # default=LOCAL_PORT,
        help="The name of the cppu"
    )
    parser.add_argument(
        "--algo",
        type=str,
        default="PPO",
        choices=["PPO", "DQN", "SAC"],
        help="The RLlib-registered algorithm to use.",
    )
    parser.add_argument(
        "--checkpoint",
        type=int,
        help="Checkpoint frequency",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        help="Out directory",
    )
    parser.add_argument(
        "--restore",
        type=int,
        help="Restore checkpoint",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Restore checkpoint",
    )


    return parser.parse_args()


def new_next(self):
    # TODO: correct logger this way?
    logger.info(f"Size of samples queue is: {self.samples_queue.qsize()}", )
    return self.samples_queue.get()


if __name__ == "__main__":

    args = get_cli_args()

    cppu_name = args.cppu
    logger = setup_logger(f'Server::{cppu_name}::PID{os.getpid()}')

    SERVER_ADDRESS = args.addr
    LOCAL_PORT = args.port
    num_workers = args.workers
    out_dir = args.outdir

    mqtt_host_url = os.getenv('MQTT_HOST_URL')
    gateway_url = os.getenv('CPPS_GATEWAY_URL')

    
    with open(args.scp) as config_file:
        simulator_config = json.load(config_file)
    with open(args.product_config_path) as config_file:
        product_config = json.load(config_file)
    with open(args.lcp) as config_file:
        learning_config = json.load(config_file)
        rllib_algorithm_config = learning_config["rllib_algorithm_config"]

    algorithm = learning_config["algorithm_class"]
    restore_checkpoint = learning_config["restore_checkpoint"]
    checkpoint_frequency = learning_config["checkpoint_frequency"]
    use_ray_tune = learning_config["use_ray_tune"]
    
    ag_sim = AgentSImulator(cppu_name=cppu_name, learning_config=learning_config, config=simulator_config,
                            products=product_config, logger=logger, mqtt_host_url=mqtt_host_url,
                            gateway_url=gateway_url)

    # Build observation space
    observation_space = ag_sim.get_observation_space()
    # obs_space_low = -1 * np.ones(37)
    # obs_space_high = np.ones(37)
    # observation_space = gym.spaces.Box(low=obs_space_low, high=obs_space_high)
    logger.info('Observation Space Server: {}'.format(observation_space))

    # Build action space 
    action_space = ag_sim.get_action_space()
    # action_space = gym.spaces.Discrete(3)

    logger.info('Action Space Server: {}'.format(action_space))
    

    def _input(io_ctx):
        # This launches a multi-threaded server that listens on the specified
        # host and port to serve policy requests and forward experiences to RLlib.
        logger.info(f'_input(io_ctx): Creating Server for {cppu_name}')
        try:
            server = PolicyServerInput(io_ctx, SERVER_ADDRESS, int(LOCAL_PORT))
            # funcType = type(server.next)
            # server.next = funcType(new_next, server, PolicyServerInput)
            logger.info(f'Server for {cppu_name} @ {LOCAL_PORT} initialized')
            return server
        except Exception as e:
            logger.info(f'Error: Server for {cppu_name} @ {LOCAL_PORT} not initialized: {str(e)}')


    ray.init()
    algo_config = algorithm
    config = (
        get_trainable_cls(algo_config).get_default_config()
        # Indicate that the Algorithm we setup here doesn't need an actual env.
        # Allow spaces to be determined by user (see below).
        .environment(
            env=None,
            observation_space=observation_space,
            action_space=action_space,
        )
        # DL framework to use.
        .framework(learning_config["framework"])  # TensorFlow or Torch
        # Use the `PolicyServerInput` to generate experiences.
        .offline_data(input_=_input)
        # Use n worker processes to listen on different ports.
        .rollouts(
            num_rollout_workers=num_workers,
            # Connectors are not compatible with the external env.
            enable_connectors=False,
        )
        .callbacks(MemoryTrackingCallbacks)
        # Disable OPE, since the rollouts are coming from online clients.
        .evaluation(off_policy_estimation_methods={})
        # Set to INFO so we'll see the server's actual address:port.
        .debugging(log_level="INFO", seed=args.seed)
    )
    # Disable RLModules because they need connectors
    config.experimental(_enable_new_api_stack=False)
    # config.rl_module(_enable_rl_module_api=False)
    # config.training(_enable_learner_api=False)

    logger.info('[Algo]: {}'.format(algorithm))
    if learning_config["action_masking"]:
        rllib_algorithm_config["policy_model_config"]["custom_model"] = ActionMaskPolicyModel
        rllib_algorithm_config["q_model_config"]["custom_model"] = ActionMaskSACQModel

    # Update config with algorithmic specific parameters
    config.update_from_dict({param: value for param, value in rllib_algorithm_config.items()})

    if use_ray_tune:
        logger.info("Using Ray Tune")
        logger.info("Ignoring restore even if previous checkpoint is provided...")

        stop = learning_config["rllib_tuner_config"]
        # Results can be visualized with Tensorboard
        tuner = tune.Tuner(
            algorithm,
            # tune_config= ,
            param_space=config,
            run_config=air.RunConfig(stop=stop, 
                                     verbose=2,
                                     name="Ray model")
        )
        publish_server_ready(logger, ag_sim, cppu_name)
        logger.info("Tuner Fit")
        results = tuner.fit()
        logger.info("Tuner Results")
        best_result = results.get_best_result()
        logger.info(f"Tuner Best Results {best_result}")
    else:
        ckp_string = f'{out_dir}/models/{cppu_name}_{algorithm}/'
        logger.info('Building config for Algorithm {}'.format(algorithm))
        algo = config.build()
        logger.info('Algorithm {} has been built'.format(algorithm))
        # algo.save(checkpoint_dir=ckp_string)
        publish_server_ready(logger, ag_sim, cppu_name)
        count = 1
        # Attempt to restore from checkpoint if possible.
        if restore_checkpoint:
            try:
                logger.info(f"Restoring from checkpoint path: {ckp_string}")
                algo.restore(ckp_string)
                logger.info('Succesfully restored')
            except Exception as e:
                logger.info(f"Unable to restore from checkpoint: {e}")

        # Serving and training loop.
        logger.info('Entering infinite training loop...')
        # save_result = algo.save(checkpoint_dir=ckp_string)
        # logger.info(f"Checkpoint saved @ {ckp_string}: {save_result}")
        while True:
            # Calls to train() will block on the configured `input` in the Trainer
            # config above (PolicyServerInput).
            results = algo.train()
            # logger.info(pretty_print(results))
            total_loss = results.get('info', {}).get('learner',{}).get('default_policy', {}).get('learner_stats',{})\
                .get('total_loss', None)
            policy_loss = results.get('info', {}).get('learner',{}).get('default_policy', {}).get('learner_stats',{})\
                .get('policy_loss', None)
            policy_entropy = results.get('info', {}).get('learner',{}).get('default_policy', {}).get('learner_stats',{})\
                .get('entropy', None)
            cpu_use = round(int(results.get('perf', {}).get('cpu_util_percent', None)), 2)
            ram_use = round(int(results.get('perf', {}).get('ram_util_percent', None)), 2)
            logger.info(f"[TRAINER_RESULTS]: total_loss: {total_loss}, policy_loss: {policy_loss},"
                        f" policy entropy: {policy_entropy}, CPU use: {cpu_use}%, RAM use: {ram_use}%")
            logger.info(pretty_print(results))
            save_result = algo.save(checkpoint_dir=ckp_string)
            logger.info(f"Checkpoint saved @ {ckp_string}: {save_result}")
