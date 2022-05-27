#! /usr/bin/env python
import rospy
import moveit_commander
from irb120_env import AbbEnv
from math import pi
import gym
import numpy as np

from stable_baselines3 import HerReplayBuffer, DDPG, DQN, SAC, TD3, PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor

from sb3_contrib import TQC

from Callback_class import SaveOnBestTrainingRewardCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback
# from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.noise import NormalActionNoise
import os
from datetime import datetime

if __name__ == '__main__':
    env = gym.make('AbbReach-v0')
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
    log_dir = "./training/"
    os.makedirs(log_dir, exist_ok=True)
    # Logs will be saved in log_dir/monitor.csv
    env = Monitor(env, log_dir)

    # Create the callback: check every 4000 steps
    callback = SaveOnBestTrainingRewardCallback(check_freq=5000, log_dir=log_dir)
    # Save a checkpoint every 4000 steps
    checkpoint_callback = CheckpointCallback(save_freq=5000, save_path='logs/',name_prefix='abb_model')

    # replay buffer
    replay_buffer_kwargs=dict(
                            online_sampling=True,
                            goal_selection_strategy='future',
                            n_sampled_goal=4
                           )


    policy_kwargs = dict(net_arch=[256,256,256], 
                            n_critics=3, 
                            n_quantiles=25
                            )

    model = TQC("MultiInputPolicy", 
                env, 
                verbose=1, 
                policy_kwargs=policy_kwargs, 
                learning_rate=0.001, 
                buffer_size=50000, 
                learning_starts=200, 
                batch_size=2048,
                tau=0.005, 
                gamma=0.6, 
                train_freq=1, 
                gradient_steps=1, 
                replay_buffer_class=HerReplayBuffer, 
                replay_buffer_kwargs=replay_buffer_kwargs, 
                target_update_interval=1, 
                target_entropy='auto', 
                top_quantiles_to_drop_per_net=2, 
                use_sde=False, 
                action_noise=action_noise,
                create_eval_env=True,
                tensorboard_log="/home/isaac/irb120_ws/src/openai_abb/src/tensorboard"
                )

    model.learn(total_timesteps=int(60000),callback=[callback,checkpoint_callback], eval_log_path="./logs/", n_eval_episodes=5,eval_freq=5000)
    model.save("/home/isaac/Desktop/model_storage/TQC_"+datetime.now().strftime("%b-%d:%H:%M:%S"))

    del model


    policy_kwargs = dict(net_arch=[256,256,256], 
                            n_critics=3, 
                            n_quantiles=25
                            )

    model = TQC("MultiInputPolicy", 
                env, 
                verbose=1, 
                policy_kwargs=policy_kwargs, 
                learning_rate=0.001, 
                buffer_size=50000, 
                learning_starts=200, 
                batch_size=2048,
                tau=0.005, 
                gamma=0.5, 
                train_freq=1, 
                gradient_steps=1, 
                replay_buffer_class=HerReplayBuffer, 
                replay_buffer_kwargs=replay_buffer_kwargs, 
                target_update_interval=1, 
                target_entropy='auto', 
                top_quantiles_to_drop_per_net=2, 
                use_sde=False, 
                action_noise=action_noise,
                create_eval_env=True,
                tensorboard_log="/home/isaac/irb120_ws/src/openai_abb/src/tensorboard"
                )

    model.learn(total_timesteps=int(60000),callback=[callback,checkpoint_callback], eval_log_path="./logs/", n_eval_episodes=5,eval_freq=5000)
    model.save("/home/isaac/Desktop/model_storage/TQC_"+datetime.now().strftime("%b-%d:%H:%M:%S"))

    del model

    policy_kwargs = dict(net_arch=[256,256,256], 
                            n_critics=3, 
                            n_quantiles=30
                            )

    model = TQC("MultiInputPolicy", 
                env, 
                verbose=1, 
                policy_kwargs=policy_kwargs, 
                learning_rate=0.001, 
                buffer_size=50000, 
                learning_starts=200, 
                batch_size=2048,
                tau=0.005, 
                gamma=0.6, 
                train_freq=1, 
                gradient_steps=1, 
                replay_buffer_class=HerReplayBuffer, 
                replay_buffer_kwargs=replay_buffer_kwargs, 
                target_update_interval=1, 
                target_entropy='auto', 
                top_quantiles_to_drop_per_net=2, 
                use_sde=False, 
                action_noise=action_noise,
                create_eval_env=True,
                tensorboard_log="/home/isaac/irb120_ws/src/openai_abb/src/tensorboard"
                )

    model.learn(total_timesteps=int(60000),callback=[callback,checkpoint_callback], eval_log_path="./logs/", n_eval_episodes=5,eval_freq=5000)
    model.save("/home/isaac/Desktop/model_storage/TQC_"+datetime.now().strftime("%b-%d:%H:%M:%S"))

    del model
    rospy.loginfo("All movements finished. Shutting down")	
    moveit_commander.roscpp_shutdown()







    # # continue training
    # loaded_model = TQC.load('/home/isaac/Desktop/model_storage/TQC_136.zip',env=env, tensorboard_log="/home/isaac/irb120_ws/src/openai_abb/src/tensorboard")
    # # load it into the loaded_model
    # loaded_model.load_replay_buffer("/home/isaac/irb120_ws/src/openai_abb/training/best_replay_buffer.pkl")

    # # now the loaded replay is not empty anymore
    # print(f"The loaded_model has {loaded_model.replay_buffer.size()} transitions in its buffer")
    # loaded_model.learn(total_timesteps=int(20000),callback=[callback,checkpoint_callback], eval_log_path="./logs/", n_eval_episodes=5,eval_freq=500)
    # # model.learn(total_timesteps=int(25000),callback=[callback,checkpoint_callback], eval_log_path="./logs/", n_eval_episodes=5,eval_freq=1000)
    # loaded_model.save("/home/isaac/Desktop/model_storage/TQC_"+datetime.now().strftime("%b-%d:%H:%M:%S"))
    # loaded_model.save_replay_buffer("best_replay_buffer_2")


    # # continue training for TQC_142
    # loaded_model = TQC.load('/home/isaac/Desktop/model_storage/TQC_May-24:18:39:21.zip',env=env, tensorboard_log="/home/isaac/irb120_ws/src/openai_abb/src/tensorboard")
    # # load it into the loaded_model
    # # loaded_model.load_replay_buffer("/home/isaac/irb120_ws/src/openai_abb/training/best_replay_buffer.pkl")

    # # now the loaded replay is not empty anymore
    # # print(f"The loaded_model has {loaded_model.replay_buffer.size()} transitions in its buffer")
    # loaded_model.learn(total_timesteps=int(20000),callback=[callback,checkpoint_callback], eval_log_path="./logs/", n_eval_episodes=5,eval_freq=500)
    # # model.learn(total_timesteps=int(25000),callback=[callback,checkpoint_callback], eval_log_path="./logs/", n_eval_episodes=5,eval_freq=1000)
    # loaded_model.save("/home/isaac/Desktop/model_storage/TQC_"+datetime.now().strftime("%b-%d:%H:%M:%S"))
    # # loaded_model.save_replay_buffer("best_replay_buffer_2")
