import numpy as np
import torch
import gym
import argparse
import os, sys

import utils
from td3 import TD3


# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment
def eval_policy(policy, env_name, seed, eval_episodes=10):
    eval_env = gym.make(env_name)
    eval_env.seed(seed + 100)

    avg_reward = 0.
    x_trj_all = []
    for _ in range(eval_episodes):
        state, done = eval_env.reset(), False
        while not done:
            action = policy.select_action(np.array(state))
            state, reward, done, _ = eval_env.step(action)
            avg_reward += reward
            x_trj_all.append(state)

    avg_reward /= eval_episodes

    print("---------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
    print("---------------------------------------")
    return x_trj_all, avg_reward


if __name__ == "__main__":

    root_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.join(root_dir, '..', '..'))
    import env
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="acrobot-v0")              # Gym environment name
    parser.add_argument("--seed", default=0, type=int)              # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--load_model", default="default")          # Model load file name, "default" uses file_name
    args = parser.parse_args()

    file_name = f"TD3_{args.env}_{args.seed}"
    env = gym.make(args.env)

    # Set seeds
    env.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0] 
    max_action = float(env.action_space.high[0])

    kwargs = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "max_action": max_action,
    }

    # Initialize policy
    policy = TD3(**kwargs)

    model_dir = os.path.join(root_dir, 'models')
    policy_file = os.path.join(model_dir, file_name) if args.load_model == "default" else args.load_model
    policy.load(policy_file)

    replay_buffer = utils.ReplayBuffer(state_dim, action_dim)
    
    # Evaluate untrained policy
    x_trj_all, _ = eval_policy(policy, args.env, args.seed)

    design = np.array([1, 2, 1, 2])
    from animate import animate
    animate(design, x_trj_all, 50)