from common.env.procgen_wrappers import *
from common import set_global_seeds, set_global_log_levels
import csv
import os, argparse
import random
from tqdm import tqdm
import config
import numpy as np

from run_utils import load_env_and_agent


def run_env(
        exp_name,
        level_seed,
        logfile=None,
        max_num_timesteps=10000,
        save_value=False,
        **kwargs):
    """
    Runs one maze level.
    returns level metrics. If logfile (csv) is supplied, metrics are also
    appended there.
    """
    if save_value:
        raise NotImplementedError

    if logfile is not None:
        append_to_csv = True

    agent = load_env_and_agent(
        exp_name=exp_name,
        env_name='maze_yellowline',
        num_envs=1,
        num_levels=1,
        start_level=level_seed,
        num_threads=1,
        **kwargs)

    obs = agent.env.reset()
    hidden_state = np.zeros((agent.n_envs, agent.storage.hidden_state_size))
    done = np.zeros(agent.n_envs)

    def log_to_csv(metrics):
        """write metrics to csv"""
        if not metrics:
            return
        column_names = ["seed", "steps", "rand_coin", "coin_collected", "inv_coin_collected", "died", "timed_out"]
        metrics = [int(m) for m in metrics]
        if append_to_csv:
            with open(logfile, "a") as f:
                w = csv.writer(f)
                if f.tell() == 0:  # write header first
                    w.writerow(column_names)
                w.writerow(metrics)

    def log_metrics(done: bool, info: dict):
        """
        When run complete, log metrics in the
        following format:
        seed, steps, randomize_goal, collected_coin, collected_inv_coin, died, timed_out
        """
        metrics = None
        if done:
            keys = ["prev_level_seed", "prev_level/total_steps", "prev_level/randomize_goal", "prev_level_complete",
                    "prev_level/invisible_coin_collected"]
            metrics = [info[key] for key in keys]
            if info["prev_level_complete"]:
                metrics.extend([False, False])
            else:
                timed_out = info["prev_level/total_steps"] > 999
                metrics.extend([not timed_out, timed_out])
        elif info["invisible_coin_collected"]:
            keys = ["level_seed", "total_steps", "randomize_goal"]
            metrics = [info[key] for key in keys]
            metrics.extend([-1, True, -1, -1])
        else:
            raise
        log_to_csv(metrics)
        return metrics

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
                # print(step, done, info)
                log_metrics(done[0], info[0])
                return
    return



if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name',         type=str, default = 'metrics', help='experiment name')
    parser.add_argument('--start_level',      type=int, default = int(0), help='start-level for environment')
    parser.add_argument('--device',           type=str, default = 'cpu', required = False, help='whether to use gpu')
    parser.add_argument('--gpu_device',       type=int, default = int(0), required = False, help = 'visible device in CUDA')
    parser.add_argument('--agent_seed',       type=int, default = random.randint(0,999999), help='Seed for pytorch')
    parser.add_argument('--log_level',        type=int, default = int(40), help='[10,20,30,40]')
    parser.add_argument('--logdir',           type=str, default = None)
    parser.add_argument('--start_level_seed', type=int, default = 0)
    parser.add_argument('--num_seeds',        type=int, default = 10)
    parser.add_argument('--random_percent',   type=int, default = 0)
    parser.add_argument('--seed_file',        type=str, help="path to text file with env seeds to run on.")
    parser.add_argument('--reset_mode',       type=str, default="inv_coin", help="Reset modes:"
                                                            "- inv_coin returns when agent gets the inv coin OR finishes the level"
                                                            "- complete returns when the agent finishes the level")

    #multi threading
    parser.add_argument('--num_threads', type=int, default=8)

    #render parameters
    parser.add_argument('--num_envs', type=int, default=1)
    parser.add_argument('--vid_dir', type=str, default=None)
    parser.add_argument('--model_file', type=str, help="Can be either a path to a model file, or an "
                                       "integer. Integer is interpreted as random_percent in training")

    args = parser.parse_args()

    # Seeds
    set_global_seeds(args.agent_seed)
    set_global_log_levels(args.log_level)

    if args.seed_file:
        print(f"Loading env seeds from {args.seed_file}")
        with open(args.seed_file, 'r') as f:
            seeds = f.read()
        seeds = [int(s) for s in seeds.split()]
    else:
        print(f"Running on env seeds {args.start_level_seed} to {args.start_level_seed + args.num_seeds}.")
        seeds = np.arange(args.num_seeds) + args.start_level_seed

    datestr = time.strftime("%Y-%m-%d_%H:%M:%S")
    logpath = os.path.join(config.results_dir, f"{args.exp_name}")
    path_to_model_file = args.model_file
    logpath = os.path.join(logpath, f"maze__" + datestr)

    os.makedirs(logpath, exist_ok=True)
    with open(os.path.join(logpath, "metadata.txt"), "a") as f:
        f.write(time.strftime("%Y-%m-%d %H:%M:%S") + f", modelfile {path_to_model_file}\n")

    logfile = os.path.join(logpath, f"metrics_agent_seed_{args.agent_seed}.csv")
    print(f"Saving metrics to {logfile}.")
    for env_seed in tqdm(seeds, disable=True):
        run_env(exp_name=args.exp_name,
            logfile=logfile,
            model_file=path_to_model_file,
            level_seed=env_seed,
            device=args.device,
            gpu_device=args.gpu_device,
            # random_percent=args.random_percent,
            # reset_mode=args.reset_mode
                )
