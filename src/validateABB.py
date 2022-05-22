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

from sb3_contrib import TQC

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
    # abb = AbbEnv()
    # abb._check_all_systems_ready()
    # abb.reset()

    # goal = [ 0.4907109,  -0.02830896,  0.43808569]

    # abb = AbbEnv()
    # print(len(abb.arm_group.get_current_joint_values()))
    # while True:
    #     while not (len(abb.arm_group.get_current_joint_values()) == 6):
    #         print("wait robot joint")  
    
    # abb.move_joint_arm(0,0.4189,0,0,0,0)
    # reward,done = abb.dense_reward(abb.get_achieved_goal(),goal)
    # print(reward)


    # for i_episode in range(20):
    #     abb.reset()
    #     print(abb.is_success(abb.achieved_goal,abb.desired_goal))
    #     info = {"is_success":abb.is_success(abb.achieved_goal,abb.desired_goal)}
    #     reward = abb.compute_reward(abb.achieved_goal,abb.desired_goal,info=info)
    #     print(reward)

    # env.reset()
    # action = env.action_space.sample()
    # observation, reward, done, info = env.step(action)
    # print(observation['desired_goal'])
    # array = observation['desired_goal']
    # env.plan_cartesian_path(array[0],array[1],array[2])
    # print(observation['achieved_goal'])

    # for i_episode in range(20):
    #     abb.reset()
    #     print(abb.desired_goal)
    #     print(abb.achieved_goal)
    #     array = abb.desired_goal
    #     abb.plan_cartesian_path(array[0],array[1],array[2])
    #     print(abb.is_success(abb.achieved_goal,abb.desired_goal))
    #     d = abb.distance(abb.achieved_goal,abb.desired_goal)
    #     print(d)

    # env = gym.make('AbbReach-v0')
    # model = TQC.load('training/best_model.zip', env=env, print_system_info=True)
    # for i_episode in range(20):
    #     observation = env.reset()
    #     for t in range(5):
    #         # print(observation)
    #         action, _states = model.predict(observation, deterministic=True)
    #         observation, reward, done, info = env.step(action)
    #         print(observation)
    #         if done:
    #             print("Episode finished after {} timesteps".format(t+1))
    #             break

    # abb.step(abb.sample_action_by_joint_values())

    env = gym.make('AbbReach-v0')
    observation = env.reset()
    action = [-1,-1,1]
    observation, reward, done, info = env.step(action)
    print(observation, reward, done, info)

    # for i_episode in range(1):
    #     observation = env.reset()
    #     for t in range(10):
    #         # print(observation)
    #         action = env.action_space.sample()
    #         observation, reward, done, info = env.step(action)
    #         if done:
    #             print("Episode finished after {} timesteps".format(t+1))
    #             break
    rospy.loginfo("All movements finished. Shutting down")	
    moveit_commander.roscpp_shutdown()