# runs many maze evals.
from common.env.procgen_wrappers import *
from common import set_global_seeds, set_global_log_levels
import copy
import csv
import os, argparse
import psutil
import random
from tqdm import tqdm
import config
import matplotlib.pyplot as plt
import numpy as np
import time

from run_utils import load_env_and_agent


AGENT_FROM = 0
AGENT_TO = 100
NUM_SEEDS = 1000


def run_env(
        exp_name,
        level_seed,
        max_num_timesteps=10000,
        save_value=False,
        use_backgrounds=False,
        save_first_obs=False,
        world_dim=5,
        obj1='red_line_diag',
        obj2='yellow_gem',
        **kwargs):
    """
    Runs one maze level.
    returns level metrics. If logfile (csv) is supplied, metrics are also
    appended there.
    """
    if save_value:
        raise NotImplementedError

    agent = load_env_and_agent(
        exp_name=exp_name,
        # env_name='maze_pure_yellowline',
        env_name='maze_colorsobjects_duel',
        distribution_mode='easy',
        # env_name='maze_yellowline_test',
        num_envs=1,
        num_levels=1,
        start_level=level_seed,
        num_threads=1,
        use_backgrounds=use_backgrounds,
        world_dim=world_dim,
        obj1=obj1,
        obj2=obj2,
        **kwargs)

    obs = agent.env.reset()
    first_obs = None
    if save_first_obs:
        first_obs = copy.deepcopy(obs)
    hidden_state = np.zeros((agent.n_envs, agent.storage.hidden_state_size))
    done = np.zeros(agent.n_envs)

    step = 0
    while step < max_num_timesteps:
        agent.policy.eval()
        for _ in range(agent.n_steps):  # = 256
            step += 1
            act, log_prob_act, value, next_hidden_state = agent.predict(obs, hidden_state, done)
            next_obs, rew, done, info = agent.env.step(act)

            agent.storage.store(obs, hidden_state, act, rew, done, info, log_prob_act, value)
            obs = next_obs
            hidden_state = next_hidden_state

            if done[0]:
                return [level_seed, step, info[0]['env_reward']], first_obs
    return [level_seed, step, 0], first_obs


def find_completed_runs(path_runs):
    completed_runs = set()
    runs = []
    if os.path.exists(path_runs):
        runs = os.listdir(path_runs)
    for run in runs:
        run_path = os.path.join(path_runs, run)
        csv_files = [f for f in os.listdir(run_path) if f.endswith('csv')]
        if len(csv_files) > 0:
            completed_runs.add(run)
    return completed_runs


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name',         type=str, default = 'metrics', help='experiment name')
    parser.add_argument('--start_level',      type=int, default = int(0), help='start-level for environment')
    parser.add_argument('--device',           type=str, default = 'cpu', required = False, help='whether to use gpu')
    parser.add_argument('--gpu_device',       type=int, default = int(0), required = False, help = 'visible device in CUDA')
    parser.add_argument('--agent_seed',       type=int, default = 42, help='Seed for pytorch')
    parser.add_argument('--log_level',        type=int, default = int(40), help='[10,20,30,40]')
    parser.add_argument('--logdir',           type=str, default = None)
    parser.add_argument('--start_level_seed', type=int, default = 100000)
    parser.add_argument('--num_seeds',        type=int, default = 10)
    parser.add_argument('--random_percent',   type=int, default = 0)
    parser.add_argument('--seed_file',        type=str, help="path to text file with env seeds to run on.")
    parser.add_argument('--reset_mode',       type=str, default="inv_coin", help="Reset modes:"
                                                            "- inv_coin returns when agent gets the inv coin OR finishes the level"
                                                            "- complete returns when the agent finishes the level")

    #multi threading
    parser.add_argument('--num_threads', type=int, default=1)

    #render parameters
    parser.add_argument('--num_envs', type=int, default=1)
    parser.add_argument('--vid_dir', type=str, default=None)
    parser.add_argument('--model_file', type=str, help="Can be either a path to a model file, or an "
                                       "integer. Integer is interpreted as random_percent in training")
    parser.add_argument('--world_dim', type=int, default=5, help='Maze grid dimension')
    parser.add_argument('--obj1', type=str, default='red_line_diag', help='Maze object 1 name')
    parser.add_argument('--obj2', type=str, default='yellow_gem', help='Maze object 2 name')
    parser.add_argument('--run_name', type=str, default='', help='run name, mainly for seeded runs')
    parser.add_argument('--use_backgrounds', action='store_true')

    args = parser.parse_args()

    set_global_log_levels(args.log_level)

    world_dim = args.world_dim
    obj1 = args.obj1
    obj2 = args.obj2

    obj1_str = obj1.replace('_', '-').replace('-diag', '')
    obj2_str = obj2.replace('_', '-').replace('-diag', '')
    path = f'logs/train/maze_pure_yellowline/maze-{world_dim}x{world_dim}-{args.run_name}/'
    path_out = f'experiments/results-1000/maze-{world_dim}x{world_dim}{"-"+args.run_name if args.run_name else ""}/{obj1_str}-{obj2_str}/'
    print(f'\nRunning experiment: {"/".join(path_out.split("/")[-3:])}, {NUM_SEEDS} seeds.')

    agent_folders = sorted(os.listdir(path))
    completed_runs = find_completed_runs(path_out)
    total_steps = 0
    first_start_time = time.time()
    for agent_idx, agent_folder in enumerate(tqdm(agent_folders[AGENT_FROM:AGENT_TO])):
        if agent_folder in completed_runs:
            continue
        model_files = sorted([f for f in os.listdir(os.path.join(path, agent_folder)) if f.endswith('.pth')],
                             key=lambda x: int(x.split('_')[1].split('.')[0]))
        for model_idx, model_file in enumerate(model_files[-1:]):
        # for model_idx, model_file in enumerate(model_files):
            start_time = time.time()
            # Seeds
            set_global_seeds(args.agent_seed)

            path_to_model_file = os.path.join(path, agent_folder, model_file)
            logpath = os.path.join(path_out, agent_folder)
            os.makedirs(logpath, exist_ok=True)

            seeds = np.arange(NUM_SEEDS) + args.start_level_seed

            logfile = os.path.join(logpath, f'{model_file[:-4]}.csv')

            outs = []
            for env_seed_idx, env_seed in enumerate(seeds):
                # save first obs of each level for first agent and just first level for other agents:
                save_first_obs = (agent_idx == 0 and AGENT_FROM == 0 or env_seed_idx == 0) and model_idx == 0
                out, obs_save = run_env(exp_name=args.exp_name,
                                        model_file=path_to_model_file,
                                        level_seed=env_seed,
                                        device=args.device,
                                        gpu_device=args.gpu_device,
                                        save_first_obs=save_first_obs,
                                        use_backgrounds=args.use_backgrounds,
                                        world_dim=world_dim,
                                        obj1=obj1,
                                        obj2=obj2,
                                        # random_percent=args.random_percent,
                                        # reset_mode=args.reset_mode
                                        )
                outs.append(out)
                if obs_save is not None:
                    first_obs_file = os.path.join(logpath, f'seed-{env_seed}.png')
                    plt.imsave(first_obs_file, np.rollaxis(obs_save[0], 0, 3))
            with open(logfile, "w") as f:
                w = csv.writer(f)
                w.writerow(['seed', 'steps', 'reward'])
                for out in outs:
                    w.writerow(out)
            steps = sum([out[1] for out in outs])
            total_steps += steps
            end_time = time.time()
            current_fps = steps / (end_time - start_time)
            fps = total_steps / (end_time - first_start_time)
            print(f'{steps} steps in {end_time - start_time:.2f} s at {current_fps:.2f} fps.')
            print(f'{total_steps} total_steps in {end_time - first_start_time:.2f} s at {fps:.2f} fps.')
            # print(psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2)  # print RAM usage by the process
            # TODO: leaks memory somewhere, can get to 40GB+ per process. Solved by buying more RAM...
