# save a video of first X frames when a certain model is evaluated.
# requires `--model_file` parameter
from common.env.procgen_wrappers import *
from common import set_global_seeds, set_global_log_levels
import copy
import csv
import cv2
import os, argparse
import psutil
import random
from tqdm import tqdm
import config
import matplotlib.pyplot as plt
import numpy as np
import skvideo.io
import time


from run_utils import load_env_and_agent

# TODO: remove these:
AGENT_FROM = 0
AGENT_TO = 100
NUM_SEEDS = 1000


def run_env(
        exp_name,
        level_seed,
        max_num_timesteps=10000,
        save_value=False,
        save_first_obs=False,
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
        use_backgrounds=False,
        world_dim=world_dim,
        obj1=obj1,
        obj2=obj2,
        **kwargs)

    frames = []
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

            frames.append(np.rollaxis(copy.deepcopy(obs[0]), 0, 3))

            if done[0]:
                # print(step, done, info, rew)
                # log_metrics(done[0], info[0])
                return [level_seed, step, info[0]['env_reward']], first_obs, frames

    return [level_seed, step, 0], first_obs, frames


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
    parser.add_argument('--vid_path', type=str, default=None)
    parser.add_argument('--model_file', type=str, help="Can be either a path to a model file, or an "
                                       "integer. Integer is interpreted as random_percent in training")
    parser.add_argument('--world_dim', type=int, default=5, help='Maze grid dimension')
    parser.add_argument('--obj1', type=str, default='red_line_diag', help='Maze object 1 name')
    parser.add_argument('--obj2', type=str, default='yellow_gem', help='Maze object 2 name')
    parser.add_argument('--video_frames', type=int, default=100)

    args = parser.parse_args()

    set_global_log_levels(args.log_level)

    world_dim = args.world_dim
    obj1 = args.obj1
    obj2 = args.obj2

    obj1_str = obj1.replace('_', '-').replace('-diag', '')
    obj2_str = obj2.replace('_', '-').replace('-diag', '')
    path = f'/home/karolis/k/goal-misgeneralization/train-procgen/logs/train/maze_pure_yellowline/maze-{world_dim}x{world_dim}-with-init-weights/'
    path_out = f'/home/karolis/k/goal-misgeneralization/train-procgen/experiments/videos/maze-{world_dim}x{world_dim}-with-init-weights/{obj1_str}-{obj2_str}/'
    print(f'\nRunning experiment: {"/".join(path_out.split("/")[-3:])}, {NUM_SEEDS} seeds.')

    agent_folders = sorted(os.listdir(path))
    total_steps = 0
    first_start_time = time.time()
    # --model_file example
    # /home/karolis/k/goal-misgeneralization/train-procgen/logs/train/maze_pure_yellowline/maze-16x16-with-init-weights/2023-09-07__12-35-49__seed_6954/model_50003968.pth

    # for agent_idx, agent_folder in enumerate(tqdm(agent_folders[AGENT_FROM:AGENT_TO])):
    #     model_files = sorted([f for f in os.listdir(os.path.join(path, agent_folder)) if f.endswith('.pth')],
    #                          key=lambda x: int(x.split('_')[1].split('.')[0]))
    #     # print(f'running {agent_folder}')
    #     for model_idx, model_file in enumerate(model_files[-1:]):
    #     # for model_idx, model_file in enumerate(model_files):
    start_time = time.time()
    # Seeds
    set_global_seeds(args.agent_seed)

    # path_to_model_file = os.path.join(path, agent_folder, model_file)
    path_to_model_file = args.model_file
    agent_folder, model_file = path_to_model_file.split('/')[-2:]
    logpath = os.path.join(path_out, agent_folder)
    os.makedirs(logpath, exist_ok=True)

    seeds = np.arange(NUM_SEEDS) + args.start_level_seed

    logfile = os.path.join(logpath, f'{model_file[:-4]}.csv')

    outs = []
    video_frames = []
    for env_seed_idx, env_seed in enumerate(seeds):
        out, obs_save, episode_frames = run_env(exp_name=args.exp_name,
                                                model_file=path_to_model_file,
                                                level_seed=env_seed,
                                                device=args.device,
                                                gpu_device=args.gpu_device,
                                                # random_percent=args.random_percent,
                                                # reset_mode=args.reset_mode
                                                )
        outs.append(out)
        video_frames += episode_frames
        if obs_save is not None:
            first_obs_file = os.path.join(logpath, f'seed-{env_seed}.png')
            plt.imsave(first_obs_file, np.rollaxis(obs_save[0], 0, 3))
        if len(video_frames) > args.video_frames:
            break
    with open(logfile, "w") as f:
        w = csv.writer(f)
        w.writerow(['seed', 'steps', 'reward'])
        for out in outs:
            w.writerow(out)

    if args.vid_path:
        vid_path = args.vid_path
    else:
        vid_path = os.path.join(logpath, 'episodes.mp4')
    video_frames = [(frame * 255).astype(np.uint8) for frame in video_frames]
    skvideo.io.vwrite(vid_path, video_frames, inputdict={'-r': '5'}, outputdict={'-pix_fmt': 'yuv420p'})

    steps = sum([out[1] for out in outs])
    total_steps += steps
    end_time = time.time()
    current_fps = steps / (end_time - start_time)
    fps = total_steps / (end_time - first_start_time)
    print(f'{steps} steps in {end_time - start_time:.2f} s at {current_fps:.2f} fps.')
    print(f'{total_steps} total_steps in {end_time - first_start_time:.2f} s at {fps:.2f} fps.')
