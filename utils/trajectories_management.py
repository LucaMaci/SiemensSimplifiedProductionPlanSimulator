import json
import numpy as np

class TrajectoryManager():
    def __init__(self, INPUT_DIR, OUTPUT_DIR, config):
        self.INPUT_DIR = INPUT_DIR
        self.OUTPUT_DIR = OUTPUT_DIR
        self.config = config
        self.n_agents = self.config['n_agents']
        self.n_products = self.config['n_products']
        self.n_episodes = self.config['n_episodes']
        self.agents_connections = {int(k): v for k, v in self.config['agents_connections'].items()} 
        self.trajectories = {}
        self.time_ordered_trajectories = {}

        # load the trajectories
        for i in range(self.n_episodes):
            with open(f'{self.INPUT_DIR}_{i}.json', 'r') as infile:
                new_run_trajectory = json.load(infile)
                self.trajectories.update(new_run_trajectory)
                self.time_ordered_trajectories[i] = new_run_trajectory
        
        # order by time the trajectory of each run
        for key in self.time_ordered_trajectories.keys():
            self.time_ordered_trajectories[key] = [item for sublist in self.time_ordered_trajectories[key].values() for item in sublist]
            self.time_ordered_trajectories[key] = sorted(self.time_ordered_trajectories[key], key=lambda x: x['time'])

    # select function to use in order to compute rewards
    # CAUTION: perform this before converting global trajectory into agents trajectories
    def compute_reward(self):
        reward_type = self.config['reward_type']
        if reward_type == 'one-step':
            self.compute_step_reward(self.config['reward_n_steps'])
        elif reward_type == 'semi-mdp':
            self.compute_semiMDP_reward()
        elif reward_type == 'negative-reward':
            self.compute_negative_reward()

    # give to each action the cumulative reward of all actions for n_steps (a step is made by all actions between one
    # transport/defer action and the followirng one)
    # CAUTION: perform this before converting global trajectory into agents trajectories
    def compute_step_reward(self, n_steps = 1):
        for episode_product in self.trajectories:
            for i in range(len(self.trajectories[episode_product]) - 1):
                reward = self.trajectories[episode_product][i]['reward']
                j = i + 1
                steps = 0
                while j < len(self.trajectories[episode_product]) and steps < n_steps:
                    if self.trajectories[episode_product][j]['action'] >= self.config['n_production_skills']:
                        steps += 1
                    if steps < n_steps:
                        reward += self.trajectories[episode_product][j]['reward']
                        j += 1
                self.trajectories[episode_product][i]['reward'] = reward

    # TO-DO: the definition of episode is changed. Check if the structure should be modified
    # give the cumulative reward of all the actions performed until the agent take again the product (if the agent
    # don't take again the product use the comulative reward until the end of the episode)
    # CAUTION: perform this before converting global trajectory into agents trajectories
    def compute_semiMDP_reward(self):
        for episode in self.trajectories:
            for i in range(len(self.trajectories[episode]) - 1):
                agent = self.trajectories[episode][i]['agent']
                reward = self.trajectories[episode][i]['reward']
                j = i + 1
                while j < len(self.trajectories[episode]) and self.trajectories[episode][j]['agent'] != agent:
                    reward += self.trajectories[episode][j]['reward']
                    j += 1
                self.trajectories[episode][i]['reward'] = reward
    
    # give at each agent for each action a reward equal to -1 times the number of products that the agent
    # saw but are not completed yet
    # CAUTION: perform this before converting global trajectory into agents trajectories
    def compute_negative_reward(self):
        for key in self.time_ordered_trajectories:
            agents_seen_products = [[0 for i in range(self.n_products)] for j in range(self.n_agents)]
            for i in range(len(self.time_ordered_trajectories[key])):
                agent = self.time_ordered_trajectories[key][i]['agent']
                if 1 in self.time_ordered_trajectories[key][i]['state']['agents_state'][agent]:
                    product = self.time_ordered_trajectories[key][i]['state']['agents_state'][agent].index(1)
                    agents_seen_products[agent][product] = 1
                for j in range(self.n_products):
                    if all(elem == 0 for elem in np.array(self.time_ordered_trajectories[key][i]['state']['products_state'][j]).flatten()):
                        agents_seen_products[agent][j] = 0
                self.time_ordered_trajectories[key][i]['reward'] = -1 * agents_seen_products[agent].count(1)

        episode__product_number = 0
        episode_number = -1
        for episode_product in self.trajectories:
            if episode__product_number % self.n_products == 0:
                episode_number += 1
            episode__product_number += 1

            for i in range(len(self.trajectories[episode_product])):
                self.trajectories[episode_product][i]['reward'] = next(item for item in self.time_ordered_trajectories[episode_number] if item['time'] == self.trajectories[episode_product][i]['time'])['reward']

    # remove from trajectories all the skills
    # CAUTION: perform this before converting global trajectory into agents trajectories
    def remove_production_skill_trajectories(self):
        for episode_product in self.trajectories:
            self.trajectories[episode_product] = [step for step in self.trajectories[episode_product] if step['action'] >= self.config['n_production_skills']]

    # remove from state all the action masks but save the ones of the current agent concerning transport and defer actions
    # CAUTION: perform this before converting global trajectory into agents trajectories
    def remove_action_masks(self):
        for episode_product in self.trajectories:
            for step in self.trajectories[episode_product]:
                step['action_mask'] = step['state']['action_mask'][f'{step["agent"]}'][6:11]
                del step['state']['action_mask']

    # this function allow to hide states of agents with a distance greater than observability grade for each agent
    # CAUTION: perform this before converting global trajectory into agents trajectories
    def set_states_observability(self):
        observability_grade = self.config['observability_grade']
        agents_state_mask = self.compute_agents_state_mask(observability_grade)
        for episode_product in self.trajectories:
            for step in self.trajectories[episode_product]:
                step['state']['agents_state'] = [lst for lst, mask in zip(step['state']['agents_state'], agents_state_mask[step['agent']]) if any(x != 0 for x in mask)]
                products_mask = np.zeros(self.n_products)
                for agent_state in step['state']['agents_state']:
                    if max(agent_state) == 1:
                        products_mask[np.argmax(agent_state)] = 1
                for i in range(len(step['state']['products_state'])):
                    if products_mask[i] == 0:
                        step['state']['products_state'][i] = np.zeros_like(step['state']['products_state'][i]).tolist()

    # extract state of current agent and product skills of his product for DISTQ algorithm
    def extract_agent_state_and_product_skills_for_DISTQ(self):
        for episode_product in self.trajectories:
            for step in self.trajectories[episode_product]:
                # add the current agent state to be able to retrieve only that if necessary
                step['state']['curr_agent_state'] = step['state']['agents_state'][step['agent']]
                # add the current product skill progress to be able to retrieve only that if necessary
                step['state']['curr_product_skills'] = step['state']['products_state'][np.argmax(step['state']['curr_agent_state'])]

    # compute a mask for each agent based on the observability grade (number of consequent neighbours to have observability on)
    def compute_agents_state_mask(self, observability_grade):
        agents_state_mask = {}
        for agent in range(self.n_agents):
            new_state = []
            for i in range(self.n_agents):
                new_state.append(np.zeros(self.n_products))
            new_state[agent] = np.ones_like(new_state[agent])

            neighbour = {}
            for i in range(observability_grade):
                if i == 0:
                    neighbour = {elem for elem in self.agents_connections[agent] if elem != None}
                else:
                    for elem in neighbour:
                        new_neighbours = {new_elem for new_elem in self.agents_connections[elem] if new_elem != None}
                        neighbour = neighbour.union(new_neighbours)

            for elem in neighbour:
                new_state[elem] = np.ones_like(new_state[elem])
                
            agents_state_mask[agent] = np.array(new_state)
        
        return agents_state_mask
    
    # convert global trajectory into agents trajectories
    def extract_agent_trajectories(self):
        filtered_episodes = {}
        for agent in range(self.n_agents):
            agent_name = f'Agent: {agent}'
            filtered_episodes[agent_name] = {}
            for key, value in self.trajectories.items():
                filtered_episodes[agent_name][key] = [filtered_val for filtered_val in value if filtered_val['agent'] == agent]

        self.trajectories = filtered_episodes

    # save as output the trajectory
    def save_trajectory(self):
        with open(self.OUTPUT_DIR, 'w') as outfile:
            json.dump(self.trajectories, outfile, indent=6)

# this function prepare input data for the distributed learning and works only
# if trajectories are global (extract_agent_trajectories has NOT been called)
def split_data_global(INPUT_DIR):
    with open(INPUT_DIR, 'r') as infile:
        trajectories = json.load(infile)
    
    t = []
    s = []
    a = []
    r = []
    s_prime = []
    absorbing = []
    sa = []
    m = []
    agents = []
    for episode_product in trajectories:
        for i in range(len(trajectories[episode_product])):
            t.append(trajectories[episode_product][i]['time'])
            # TO-DO: check what is better
            #s.append(flatten_dict_values(trajectories[episode][i]['state']))
            s.append(trajectories[episode_product][i]['state'])
            a.append(trajectories[episode_product][i]['action'])
            r.append(trajectories[episode_product][i]['reward'])
            # TO-DO: fix absorbing, for now it works only for the last agent that sees each product
            # but should work for the last time that each agent see that product
            absorbing.append(1 if i == (len(trajectories[episode_product]) - 1) else 0)
            # TO-DO: check if you have to change depending on what decided for s
            temp_sa = flatten_dict_values(trajectories[episode_product][i]['state'])
            temp_sa.append(trajectories[episode_product][i]['action'])
            sa.append(temp_sa)
            m.append(trajectories[episode_product][i]['action_mask'])
            agents.append(trajectories[episode_product][i]['agent'])

    # TO-DO: fix s_prime
    s_prime = s[1:]
    s_prime.append(s[-1])
    return t, s, a, r, s_prime, absorbing, sa, m, agents

# this function prepare input data for the selected agent learning but works only
# if extract_agent_trajectories function has been called
def split_data_single_agent(INPUT_DIR, agent):
    with open(INPUT_DIR, 'r') as infile:
        trajectories = json.load(infile)
    
    trajectory = trajectories[f'Agent: {agent}']
    t = []
    s = []
    a = []
    r = []
    s_prime = []
    absorbing = []
    sa = []
    m = []
    for episode_product in trajectory:
        for i in range(len(trajectory[episode_product])):
            t.append(trajectory[episode_product][i]['time'])
            s.append(flatten_dict_values(trajectory[episode_product][i]['state']))
            a.append(trajectory[episode_product][i]['action'])
            r.append(trajectory[episode_product][i]['reward'])
            absorbing.append(1 if i == (len(trajectory[episode_product]) - 1) else 0)
            temp_sa = flatten_dict_values(trajectory[episode_product][i]['state'])
            temp_sa.append(trajectory[episode_product][i]['action'])
            sa.append(temp_sa)
            m.append(trajectory[episode_product][i]['action_mask'])

    # TO-DO: fix s_prime
    s_prime = s[1:]
    s_prime.append(s[-1])
    #print(len(s))
    #print(len(sa))
    #print(f'len of s: {len(s[0])}')
    #print(s[0])
    return np.array(t), np.array(s), np.array(a), np.array(r), np.array(s_prime), np.array(absorbing), np.array(sa), np.array(m)

def flatten_dict_values(d):
    flattened_values = []
    if isinstance(d, dict):
        for value in d.values():
            flattened_values.extend(flatten_dict_values(value))
    elif isinstance(d, list):
        for item in d:
            flattened_values.extend(flatten_dict_values(item))
    elif isinstance(d, np.ndarray):
        for item in d:
            flattened_values.extend(flatten_dict_values(item))
    else:
        flattened_values.append(d)
    return flattened_values



    


