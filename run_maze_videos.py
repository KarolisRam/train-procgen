# save a video of first X frames when a certain model is evaluated.
# requires `--model_file` parameter
import argparse
import copy
import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
import time

from common import set_global_seeds, set_global_log_levels
from run_utils import load_env_and_agent

NUM_SEEDS = 1000


def run_env(
        exp_name,
        level_seed,
        max_num_timesteps=10000,
        save_value=False,
        save_first_obs=False,
        use_backgrounds=False,
        env_name='maze_colorsobjects_duel',
        world_dim=5,
        obj1='red_line_diag',
        obj2='yellow_gem',
        **kwargs):
    """
    Runs one maze level.
    returns level metrics and frames.
    """
    if save_value:
        raise NotImplementedError

    agent = load_env_and_agent(
        exp_name=exp_name,
        env_name=env_name,
        distribution_mode='easy',
        num_envs=1,
        num_levels=1,
        start_level=level_seed,
        num_threads=1,
        use_backgrounds=use_backgrounds,
        world_dim=world_dim,
        obj1=obj1,
        obj2=obj2,
        **kwargs)

    frames = []
    obs = agent.env.reset()
    frames.append(np.rollaxis(copy.deepcopy(obs[0]), 0, 3))
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
                return [level_seed, step, info[0]['env_reward']], first_obs, frames[:-1]

    return [level_seed, step, 0], first_obs, frames[:-1]


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
    parser.add_argument('--vid_path', type=str, required=True, default="videos/episodes.avi",
                        help="output video path. use avi for lossless")
    parser.add_argument('--model_file', type=str, required=True, help="path to a model file")
    parser.add_argument('--world_dim', type=int, default=5, help='Maze grid dimension')
    parser.add_argument('--obj1', type=str, default='', help='Maze object 1 name')
    parser.add_argument('--obj2', type=str, default='', help='Maze object 2 name')
    parser.add_argument('--video_frames', type=int, default=100, help='Number of first frames to save')
    parser.add_argument('--use_backgrounds', action='store_true')
    parser.add_argument('--env_name', type=str, default='maze_colorsobjects_duel')

    args = parser.parse_args()

    set_global_log_levels(args.log_level)

    world_dim = args.world_dim
    obj1 = args.obj1
    obj2 = args.obj2

    total_steps = 0
    first_start_time = time.time()
    start_time = time.time()
    # Seeds
    set_global_seeds(args.agent_seed)

    path_to_model_file = args.model_file
    agent_folder, model_file = path_to_model_file.split('/')[-2:]

    seeds = np.arange(NUM_SEEDS) + args.start_level_seed

    outs = []
    video_frames = []
    for env_seed_idx, env_seed in enumerate(seeds):
        out, obs_save, episode_frames = run_env(exp_name=args.exp_name,
                                                model_file=path_to_model_file,
                                                level_seed=env_seed,
                                                device=args.device,
                                                gpu_device=args.gpu_device,
                                                use_backgrounds=args.use_backgrounds,
                                                env_name=args.env_name,
                                                world_dim=world_dim,
                                                obj1=obj1,
                                                obj2=obj2,
                                                # random_percent=args.random_percent,
                                                # reset_mode=args.reset_mode
                                                )
        outs.append(out)
        video_frames += episode_frames
        if len(video_frames) > args.video_frames:
            break

    vid_path = args.vid_path
    os.makedirs(os.path.dirname(vid_path), exist_ok=True)
    video_frames = [(frame * 255).astype(np.uint8) for frame in video_frames][:args.video_frames]  # rescale
    video_frames = [frame[:, :, ::-1] for frame in video_frames]  # RGB->BGR
    height, width, layers = video_frames[0].shape

    out = cv2.VideoWriter(vid_path, cv2.VideoWriter_fourcc(*'FFV1'), 5.0, (width, height))  # lossless codec

    for frame in video_frames:
        out.write(frame)
    out.release()

    steps = sum([out[1] for out in outs])
    total_steps += steps
    end_time = time.time()
    current_fps = steps / (end_time - start_time)
    fps = total_steps / (end_time - first_start_time)
