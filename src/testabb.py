#! /usr/bin/env python
import rospy
import moveit_commander
from irb120_env import AbbEnv
from math import pi
# from utils.util import ABB
# from openai_ros.openai_ros_common import ROSLauncher
from gazebo_msgs.msg import ModelState 
from gazebo_msgs.srv import SetModelState
import geometry_msgs.msg
import copy
import moveit_msgs.msg
import gym


from stable_baselines3 import HerReplayBuffer, DDPG, DQN, SAC, TD3, PPO
from stable_baselines3.common.evaluation import evaluate_policy




if __name__ == '__main__':
    env = gym.make('AbbReach-v0')

    # # Instantiate the agent
    # policy_kwargs = dict(net_arch=[64, 64])

    # replay_buffer_kwargs=dict(
    # online_sampling=True,
    # goal_selection_strategy='future',
    # n_sampled_goal=4
    # )
    # model = SAC("MultiInputPolicy", 
    #             env, 
    #             verbose=1, 
    #             policy_kwargs=policy_kwargs, 
    #             ent_coef= 'auto',
    #             learning_rate=0.001, 
    #             buffer_size=1000000, 
    #             learning_starts=1000, 
    #             batch_size=256, 
    #             tau=0.005, 
    #             gamma=0.95, 
    #             replay_buffer_class=HerReplayBuffer, 
    #             replay_buffer_kwargs=replay_buffer_kwargs, 
    #             target_entropy='auto', 
    #             use_sde=False, 
    #             sde_sample_freq=- 1, 
    #             use_sde_at_warmup=False, 
    #             create_eval_env=False,
    #             tensorboard_log="/home/isaac/testOpenAi/src/openai_ros/openai_ros/src/openai_ros/robot_envs/tensorboard"
    #             )

    # mean_reward, std_reward = evaluate_policy(model, env)

    # print(f"Mean reward = {mean_reward:.2f} +/- {std_reward:.2f}")
    # env.reset()
    # observation, reward, done, info = env.step(env.action_space.sample())
    # print("obser:", info)
    # model.learn(total_timesteps=int(10000))
    for i_episode in range(20):
        observation = env.reset()
        for t in range(20):
            # print(observation)
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
            if done:
                print("Episode finished after {} timesteps".format(t+1))
                break

    rospy.loginfo("All movements finished. Shutting down")	
    moveit_commander.roscpp_shutdown()