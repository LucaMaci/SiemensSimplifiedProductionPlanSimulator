import numpy as np
import json
import copy

class OutputGenerator():
    def __init__(self, config, run, model_path):
        self.output_log = config['output_log']
        self.export_trajectories = config['export_trajectories']

        self.OUTPUT_PATH = config['output_path']
        self.TRAJECTORY_PATH = config['trajectory_path']

        self.run = run
        self.n_products = config['n_products']
        self.n_agents = config['n_agents']
        self.test_model = config['test_model']
        if self.test_model:
            self.file_log_name = f"{model_path}/test_logs.txt"
            self.performance_log_name = f"{model_path}/test_performance.txt"
            self.last_episode = config['test_model_n_episodes'] - 1
        else:
            self.file_log_name = f"{model_path}/training_logs.txt"
            self.performance_log_name = f"{model_path}/training_performance.txt"
            self.last_episode = config['n_episodes'] - 1
    
        self.n_production_skills = config['n_production_skills']
        self.nothing_action = self.n_production_skills + 5

    def generate_outputs(self, input_trajectory, episode):
        if self.output_log:
            self.generate_episode_output_file(input_trajectory, episode)
        if self.export_trajectories:
            self.generate_episode_export_trajectories_file(input_trajectory, episode)

    # create output
    def generate_episode_output_file(self, input_trajectory, episode):
        with open(f"{self.OUTPUT_PATH}_run{self.run}_{episode}.txt", 'w') as file:
            for actual_step in input_trajectory:
                file.write(f"***************Episode{episode}***************\n")
                file.write(f"***************Step{actual_step['step']}***************\n")
                file.write(f"Time: {actual_step['old_state']['time']}, Current agent: {actual_step['old_state']['current_agent']}\n")
                file.write(f"Agents busy: {actual_step['old_state']['agents_busy']}\n")
                file.write(f"Agents state: \n{actual_step['old_state']['agents_state']}\n")
                file.write(f"Products state: \n{actual_step['old_state']['products_state']}\n")
                file.write(f"Action mask: \n{actual_step['old_state']['action_mask']}\n")
                file.write(f"Step: {actual_step['step']}, Action: {actual_step['action']}, Reward: {actual_step['reward']}\n\n")

    # create trajectory
    def generate_episode_export_trajectories_file(self, input_trajectory, episode):
        trajectories = {f'Episode {episode}, Product {product}': [] for product in range(self.n_products)}
                
        for actual_step in input_trajectory:
            if actual_step['action'] != self.nothing_action:
                state_to_save = copy.deepcopy(actual_step['old_state'])
                del state_to_save['time'], state_to_save['current_agent'], state_to_save['agents_busy']
                state_to_save['action_mask'] = {key: value.tolist() if isinstance(value, np.ndarray) else value for key, value in state_to_save['action_mask'].items()}
                state_to_save = {key: value.tolist() if isinstance(value, np.ndarray) else value for key, value in state_to_save.items()}
                trajectory_update = {'time': int(actual_step['old_state']['time']), 'agent': int(actual_step['old_state']['current_agent']), 'state': state_to_save, 'action': int(actual_step['action']), 'reward': int(actual_step['reward'])}
                current_product = np.argmax(actual_step['new_state']['agents_state'][actual_step['old_state']['current_agent']]) if actual_step['action'] == 0 else np.argmax(actual_step['old_state']['agents_state'][actual_step['old_state']['current_agent']])
                trajectories[f"Episode {episode}, Product {current_product}"].append(trajectory_update)

        with open(f"{self.TRAJECTORY_PATH}_run{self.run}_{episode}.json", 'w') as outfile:
            json.dump(trajectories, outfile, indent=6)
                
    # create logs
    def generate_log_file(self):
        with open(self.file_log_name, 'w') as file:
            file.write(f"")

    def start_new_episode_log(self, episode):
        with open(self.file_log_name, 'a') as file:
            file.write(f"Starting episode {episode+1}\n")

    def pick_up_new_product_log(self, agent, product):
        with open(self.file_log_name, 'a') as file:
            file.write(f"Product {product} picked up by agent {agent}\n")

    def production_skill_log(self, action, product, agent):
        with open(self.file_log_name, 'a') as file:
            file.write(f"Production skill {action} performed on product {product} by agent {agent}\n")

    def transfer_action_log(self, source_agent, product, dest_agent, action):
        with open(self.file_log_name, 'a') as file:
            file.write(f"Agent {source_agent} transfer product {product} to agent {dest_agent} (action: {action})\n")

    def end_episode_log(self, episode):
        with open(self.file_log_name, 'a') as file:
            file.write(f"Episode {episode+1} ended\n")

    def generate_performance_log(self, performance, episode):
        if episode == self.last_episode:
            with open(self.performance_log_name, 'w') as file:
                episodes_time_to_complete = [performance[i]['episode_duration'] for i in performance]
                file.write(f"Min time to complete: {np.min(episodes_time_to_complete)}\n")
                file.write(f"Avg time to complete: {np.average(episodes_time_to_complete)}\n")
                file.write(f"Max time to complete: {np.max(episodes_time_to_complete)}\n")
                file.write(f"Variance of time to complete: {np.var(episodes_time_to_complete)}\n")
                for j in range(self.n_agents):
                    file.write(f"Agent {j} mean reward: {np.nanmean([performance[i][j]['mean_reward'] for i in performance])}\n")
                file.write(f"\n")
                for i in performance:
                    file.write(f"****Episode {i+1}****\n")
                    file.write(f"Mean reward:\n")
                    for j in range(self.n_agents):
                        file.write(f"Agent {j}: {performance[i][j]['mean_reward']}\n")
                    file.write(f"Time to complete: {performance[i]['episode_duration']}\n\n")
