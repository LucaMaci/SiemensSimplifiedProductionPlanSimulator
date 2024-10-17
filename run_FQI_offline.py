import json
import numpy as np
from utils.trajectories_management import split_data_single_agent
from utils.learning_policies_utils import initialize_agents

algorithm = "FQI"
n_episodes = 50
n_agents = 9
reward_type = "reward1"

learning_agents = initialize_agents(n_agents, algorithm, n_episodes, reward_type)
for agent in learning_agents:
    agent.iter()

_, _, _, r, s_prime, absorbing, sa, _ = split_data_single_agent("output/export_trajectories_FQI_NEW_TEST_POSTPROCESSED.json", 2)
example_state = sa[10][:-1]
mask = [1, 1, 1, 1, 1]
action = learning_agents[2].select_action(example_state, mask)
print(f'action selected: {action}')

for agent in learning_agents:
    agent.save()

'''INPUT_DIR = "output/export_trajectories6_POSTPROCESSED.json"
CONFIG_PATH = "config/FQI_config.json"

if __name__ == '__main__':
    with open(CONFIG_PATH) as config_file:
        config = json.load(config_file)

    n_agents = config['n_agents']
    actions = config['available_actions']
    agents_connections = {int(k): v for k, v in config['agents_connections'].items()}

    regressor_params = config['regressor_params']
    
    max_iterations = config['max_iterations']
    batch_size = config['batch_size']
    n_runs = config['n_runs']
    n_jobs = config['n_jobs']
    fit_params = config['fit_params']

    # FQI
    # TO-DO: before FQI fix absorbing (see trajectories_management)
    pi = []
    mdps = []
    agents = []
    for i in range(n_agents):
        _, _, _, r, s_prime, absorbing, sa, _ = split_data_single_agent(INPUT_DIR, i)

        agent_actions = [action for action, mask in zip(actions[:-1], agents_connections[i]) if mask != None]
        agent_actions.append(actions[-1])
        print(agent_actions)

        pi.append(EpsilonGreedy(agent_actions, ZeroQ(), 0))

        mdps.append(MDP(len(s_prime[0]), agent_actions))

        agents.append(FQI(mdps[i], pi[i], verbose = True, actions = agent_actions,
                        batch_size = batch_size, max_iterations = max_iterations,
                        regressor_type = ExtraTreesRegressor, **regressor_params))

        agents[i].reset()

        for _ in range(max_iterations):
            agents[i]._iter(sa, r, s_prime, absorbing, **fit_params)

    example_state = sa[10][:-1]
    print(f'example_state: {example_state}')
    chosen_action = agents[8]._policy.sample_action(example_state)
    print('action {} has been chosen for state {}'.format(chosen_action, example_state))

    #print(pi[8].Q.values(sa))

    agents[8]._policy.Q.save("prova")
    agents[8]._policy.Q.load("prova")

    chosen_action = agents[8]._policy.sample_action(example_state)
    print('action {} has been chosen for state {}'.format(chosen_action, example_state))

'''