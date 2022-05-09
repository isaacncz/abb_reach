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
import numpy as np



if __name__ == '__main__':
    # # abb._check_all_systems_ready()
    # # print(abb.get_achieved_goal())
    # # [0.23269473 0.26516924 0.29481993]
    # # [0.61221261 0.0650267  0.2401592 ]
    # # xyz = np.array([0.002694, 0.005169, 0.294819])
    # xyz = np.array([0.4, -0.35,  0.2 ])
    # # plan = abb.go_to_pose_goal(x=xyz[0],y=xyz[1],z=xyz[2])

    # action_plan = abb.plan_cartesian_path(x=xyz[0],y=xyz[1],z=xyz[2])
    abb = AbbEnv()
    abb._check_all_systems_ready()
    abb.step(abb.sample_action_by_joint_values())
    # env = gym.make('AbbReach-v0')
    # for i_episode in range(20):
    #     observation = env.reset()
    #     for t in range(20):
    #         # print(observation)
    #         action = env.action_space.sample()
    #         observation, reward, done, info = env.step(action)
    #         if done:
    #             print("Episode finished after {} timesteps".format(t+1))
    #             break
    rospy.loginfo("All movements finished. Shutting down")	
    moveit_commander.roscpp_shutdown()