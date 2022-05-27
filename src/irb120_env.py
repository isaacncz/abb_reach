import numpy as np
import rospy
import sys
import moveit_commander
import moveit_msgs.msg

from moveit_msgs.srv import GetPositionFK, GetPositionFKRequest
import geometry_msgs.msg
import copy
from std_msgs.msg import Header
from moveit_commander.conversions import pose_to_list
from sensor_msgs.msg import JointState

from math import pi
import math
import os
from gym import spaces
import gym

# moveit reference
# http://docs.ros.org/en/jade/api/moveit_commander/html/classmoveit__commander_1_1move__group_1_1MoveGroupCommander.html#a6a4440597144e033cd1749e0e00716ca

# distance
from typing import Any, Dict, Optional, Tuple, Union
# generate random number
import random

from visualization_msgs.msg import Marker

from gym.envs.registration import register

reg = register(
            id="AbbReach-v0",
            entry_point='irb120_env:AbbEnv',
            max_episode_steps=50,
        )


class AbbEnv(gym.Env):
    """Superclass for all Robot environments.
    """

    def __init__(self):
        """Initializes a new Robot environment.
        """

        # moveit config
        self.controllers_list = []
        self.robot_name_space = ""
        self.reset_controls = False

        # Init stuff
        moveit_commander.roscpp_initialize(sys.argv)
        rospy.init_node('moving_irb120', anonymous=True)
        # Variables that we give through the constructor.
        # rospy.logdebug("Start abbenv INIT...")
        self.robot = moveit_commander.RobotCommander()
        self.scene = moveit_commander.PlanningSceneInterface()
        self.arm_group = moveit_commander.MoveGroupCommander("manipulator")
        self.arm_group.set_planner_id("RRTConnectkConfigDefault")
        # self.arm_group.set_planner_id("TRAC_IKKinematicsPlugin")
        # rospy.logdebug("end var INIT...")

        # RL 
        self.distance_threshold = 0.05
        self.reward_type = "dense"

        self.desired_goal = np.zeros(3)
        self.achieved_goal = np.zeros(3)
        self.action_space = spaces.Box(-1, 1, shape=(3,), dtype=np.float32)

        obs = self._get_obs()
        # observation_shape = obs["observation"].shape
        # achieved_goal_shape = obs["achieved_goal"].shape
        # desired_goal_shape = obs["achieved_goal"].shape
        # print(observation_shape,achieved_goal_shape,desired_goal_shape)
        self.observation_space = spaces.Dict(
            dict(
                desired_goal=spaces.Box(
                    low=-1, high=1, shape=obs["achieved_goal"].shape, dtype="float32"
                ),
                achieved_goal=spaces.Box(
                    low=-1, high=1, shape=obs["achieved_goal"].shape, dtype="float32"
                ),
                observation=spaces.Box(
                    low=-1, high=1, shape=obs["observation"].shape, dtype="float32"
                ),
            )
        )
        # print(self.observation_space)


        # Misc variables
        self.box_name = "box"

        # Publish trajectory in RViz
        self.display_trajectory_publisher = rospy.Publisher('/move_group/display_planned_path',
                                                    moveit_msgs.msg.DisplayTrajectory,
                                                    queue_size=20)

        super(AbbEnv, self).__init__()

    def get_achieved_goal(self) -> np.ndarray:
        rospy.wait_for_service('compute_fk')
        try:
            moveit_fk = rospy.ServiceProxy('compute_fk',GetPositionFK)
        except rospy.ServiceException as e:
            rospy.logerr("Service call failed:",e)
        rqst = GetPositionFKRequest()

        rqst.header.frame_id = 'base_link'
        rqst.fk_link_names = ['link_6']
        rqst.robot_state.joint_state.name = []
        rqst.robot_state.joint_state.position = []
        i = 1
        lastPosition = [0, 0, 0]
        joint_goal = self.arm_group.get_current_joint_values()
        while not (len(joint_goal) == 6):
            print("[WARN] Waiting for robot joint values")
            joint_goal = self.arm_group.get_current_joint_values()

        if (len(joint_goal) == 6):
            while (i<7):
                rqst.robot_state.joint_state.name.append('joint_'+str(i))
                rqst.robot_state.joint_state.position.append(joint_goal[i-1])
                i+=1
            res = moveit_fk(rqst)
            # print(res.pose_stamped[0].pose.position)
            position = [0, 0, 0]
            
            position[0] = res.pose_stamped[0].pose.position.x
            position[1] = res.pose_stamped[0].pose.position.y
            position[2] = res.pose_stamped[0].pose.position.z
            position = np.array(position)
            lastPosition = position
            self.achieved_goal = position
            # print('self.achieved_goal',self.achieved_goal)
            return position
        print(lastPosition) 
        return lastPosition

    def plan_cartesian_path(self,x,y,z):
        waypoints = []
        scale = 1
        wpose = self.arm_group.get_current_pose().pose
        wpose.position.x = x
        wpose.position.y = y
        wpose.position.z = z

        waypoints.append(copy.deepcopy(wpose))

        (plan, fraction) = self.arm_group.compute_cartesian_path(
            waypoints, 0.01, 0.0  # waypoints to follow  # eef_step
        )  # jump_threshold
        self.arm_group.execute(plan, wait=True)
    
    def get_joint_values(self):
        print("Get the current configuration",self.arm_group.get_current_joint_values())
        print("Get the current pose",self.arm_group.get_current_pose())
        return self.arm_group.get_current_pose().pose


    def sample_action_by_joint_values(self):
        joint_goal = self.arm_group.get_current_joint_values()        
        self.arm_group.set_goal_joint_tolerance(0.001)
        joint_goal[0] = random.uniform(-0.15, 0.15)
        joint_goal[1] = random.uniform(-0.15, 0.15)
        joint_goal[2] = random.uniform(-0.15, 0.15)
        joint_goal[3] = random.uniform(-0.15, 0.15)
        joint_goal[4] = random.uniform(-0.15, 0.15)
        joint_goal[5] = random.uniform(-0.15, 0.15)
        return joint_goal

    # Forward Kinematics (FK): move the arm by axis values
    def move_joint_arm(self,joint_0,joint_1,joint_2,joint_3,joint_4,joint_5):
        joint_goal = self.arm_group.get_current_joint_values()
        self.arm_group.set_goal_joint_tolerance(0.001)
        # print("joint goal:",joint_goal)
        convertToNumpy = np.array(joint_goal)
        # print("joint numpy:",convertToNumpy)
        joint_goal[0] = joint_0
        joint_goal[1] = joint_1
        joint_goal[2] = joint_2
        joint_goal[3] = joint_3
        joint_goal[4] = joint_4
        joint_goal[5] = joint_5

        self.arm_group.go(joint_goal, wait=True)
        self.arm_group.stop() # To guarantee no residual movement
        # self.arm_group.clear_pose_targets()

    def _get_obs(self) -> Dict[str, np.ndarray]:
        """
        Here we define what sensor data of our robots observations
        :return: observations
        """
        current_pose = self.arm_group.get_current_pose().pose
        # print("current pose:",current_pose)
        achieved_goal = self.achieved_goal.copy()
        desired_goal = self.desired_goal.copy()

        xyz_distance = np.subtract(desired_goal, achieved_goal)

        d = self.distance(achieved_goal, desired_goal)
        is_success = self.is_success(self.achieved_goal,self.desired_goal)
        arrayOfObs = [    
            current_pose.position.x,
            current_pose.position.y,
            current_pose.position.z,
            # xyz_distance[0],
            # xyz_distance[1],
            # xyz_distance[2],
            d
            # is_success
        ]
        currentPoseArray = np.array(arrayOfObs)
        # print("Current pose:",currentPoseArray) # to normalize this value
        # print("Current pose:",*currentPoseArray)
        # joint_angles = self.arm_group.get_joint_value_target() # [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        # print("joint_angles: ",joint_angles)

        # return currentPoseArray.copy()

        return {
            "observation": currentPoseArray.copy(),
            # "observation": self.get_achieved_goal(),
            "achieved_goal": self.get_achieved_goal(),
            "desired_goal": self.desired_goal
        }


    # completed _get_joints
    def _get_joints(self):
        return self.arm_group.get_active_joints() # ['joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6']

    # completed _check_all_systems_ready
    def _check_all_systems_ready(self):
        """
        Checks that all the sensors, publishers and other simulation systems are
        operational.
        """
        self._check_joint_states_ready()
        return True

    # completed _check_joint_states_ready
    def _check_joint_states_ready(self):
        self.joint_states = None
        # rospy.logdebug("Waiting for /joint_states to be READY...")
        while self.joint_states is None and not rospy.is_shutdown():
            try:
                self.joint_states = self.robot.get_current_state()
                # rospy.logdebug("Current /joint_states READY=>")
            except:
                rospy.logerr("Current /joint_states not ready yet, retrying for getting joint_states")
        return self.joint_states
    
    # Methods that the TrainingEnvironment will need to define here as virtual
    # because they will be used in RobotGazeboEnv GrandParentClass and defined in the
    # TrainingEnvironment.
    # ----------------------------
    def _set_init_pose(self):
        """Sets the Robot in its init pose
        """
        raise NotImplementedError()
    
    
    def _init_env_variables(self):
        """Inits variables needed to be initialised each time we reset at the start
        of an episode.
        """
        raise NotImplementedError()

    def _compute_reward(self, observations, done):
        """Calculates the reward to give based on the observations given.
        """
        raise NotImplementedError()

    def _set_action(self, action):
        """Applies the given action to the simulation.
        """
        raise NotImplementedError()

    # def _get_obs(self):
    #     raise NotImplementedError()

    def _is_done(self, observations):
        """Checks if episode done based on observations given.
        """
        raise NotImplementedError()
    
    def wait_time_for_execute_movement(self):
        """
        Because this Parrot Drone position is global, we really dont have
        a way to know if its moving in the direction desired, because it would need
        to evaluate the diference in position and speed on the local reference.
        """
        rospy.sleep(1.0)


    # from panda gym
    # https://github.com/qgallouedec/panda-gym/blob/master/panda_gym/envs/tasks/reach.py
    def reset(self) -> Dict[str, np.ndarray]:
        self.desired_goal = self._sample_goal() 

        self.add_marker(goal=self.desired_goal) # add box to rviz for visualization
        self.move_joint_arm(0,0,0,0,0,0) #g o back to neutral position
        self.wait_time_for_execute_movement()
        return self._get_obs()

    def _get_action_space(self)-> gym.spaces.Box:
        """Get environment action space.

        Returns:
            gym.spaces: Gym action space object.

        """
        return self.action_space 


    def step(self, action) -> Tuple[np.array, float, bool, dict]:
        action = action.copy()
        if type(action) == list: action = np.array(action)
     
        joint_goal = self.arm_group.get_current_joint_values()
        while not (len(joint_goal) == 6):
            print("[WARN] Waiting for robot joint values")
            joint_goal = self.arm_group.get_current_joint_values()

        self.arm_group.set_goal_joint_tolerance(0.001)
        # scale = 1
        # joint_goal[0] = joint_goal[0] + (action[0] * scale)
        # joint_goal[1] = joint_goal[1] + (action[1] * scale)
        # joint_goal[2] = joint_goal[2] + (action[2] * scale)
        joint_goal[0] = action[0] * math.radians(70) 
        joint_goal[1] = action[1] * math.radians(50) 
        joint_goal[2] = action[2] * math.radians(50) 
        joint_goal[3] = 0
        joint_goal[4] = 0
        joint_goal[5] = 0
        # joint_goal[3] = action[3] * 2.79253 * 0.1
        # joint_goal[4] = action[4] * 2.09440 * 0.1
        # joint_goal[5] = action[5] * 6.98132 * 0.1

        self.arm_group.go(joint_goal, wait=True)
        self.arm_group.stop() # To guarantee no residual movement

        # Assign reward
        reward = 0
        done = False
        info = {}
        state = self._get_obs()

        position = self.get_achieved_goal() # compute the reward before returning the info
        # print("Position:",position)
        # print("compute reward here:",self.compute_reward(self.achieved_goal,self.desired_goal))

        info = {"is_success":self.is_success(self.achieved_goal,self.desired_goal)}
        reward, done = self.dense_reward(self.achieved_goal,self.desired_goal)

        # if joint_goal[0] > pi/2 or joint_goal[0] < -pi/2: # if more/less than +-90 degree
        #     done = True 
        #     reward = -50
        #     print("joint 1 limit exceeds")
        # if joint_goal[1] > 1.047 or joint_goal[1] < -1.047: # if more/less than +-60 degree
        #     done = True 
        #     reward = -50
        #     print("joint 2 limit exceeds")
        # if joint_goal[2] > 1.047 or joint_goal[2] < -1.047: # if more/less than +-60 degree
        #     done = True 
        #     reward = -50
        #     print("joint 3 limit exceeds")

        if done:
            print("Episode finished")
        # info = {"is_success": done}
        # info = {"is_success": self.task.is_success(obs["achieved_goal"], self.task.get_goal())}
        # print("Reward:",reward," info:",info)
        # reward, done, info = self.reward(rs_state=rs_state, action=action)

        # if self.rs_state_to_info: info['rs_state'] = self.rs_state
        return state, reward, done, info

    def _sample_goal(self) -> np.ndarray:
        """Randomize goal."""
        x = random.uniform(0.3, 0.4)
        y = random.uniform(-0.35, 0.35)
        z = random.uniform(0.25, 0.35)

        goal = np.array([x,y,z])

        print("sample goal (xyz):",goal)
        return goal

    def is_success(self, achieved_goal: np.ndarray, desired_goal: np.ndarray) -> Union[np.ndarray, float]:
        d = self.distance(achieved_goal, desired_goal)
        # return 1-d 
        # print(np.array(d <= self.distance_threshold, dtype=np.float64))
        return np.array(d <= self.distance_threshold, dtype=np.float64)

    def distance(self,a: np.ndarray, b: np.ndarray) -> Union[float, np.ndarray]:
        """Compute the distance between two array. This function is vectorized.
        Args:
            a (np.ndarray): First array.
            b (np.ndarray): Second array.
        Returns:
            Union[float, np.ndarray]: The distance between the arrays.
        """
        assert a.shape == b.shape
        return np.linalg.norm(a - b, axis=-1)

    def dense_reward(self, achieved_goal, desired_goal):
        """
        Description:
            Generates a dense reward for the given states.
        Args:
            desired_goal ([type=np.float32, size=(desired_goal.shape,)]): Desired Goal Position
            achieved_goal ([type]): [description]
        Returns:
            [np.float32]: Dense Reward
        """
        done = False

        d = np.linalg.norm(desired_goal - achieved_goal) 
        if d <= self.distance_threshold:
            reward = 1
        elif d > 0.2:
            reward = -1
        else: 
            # reward = 1-d
            reward = -np.power(d,0.9)
            # reward = 1-np.power(d,0.05)

        return reward, done

    def compute_reward(self, achieved_goal, desired_goal, info: Dict[str, Any]) -> Union[np.ndarray, float]:

        d = np.linalg.norm(desired_goal - achieved_goal) 
        # if d > self.distance_threshold:
        #     reward = self.remapReward(d)
        # else: 
        #     reward = 1

        if d <= self.distance_threshold:
            reward = 1
        elif d > 0.2:
            reward = -1
        else: 
            # reward = 1-d
            # reward = 1-np.power(d,0.05)
            reward = -np.power(d,0.9)
            # reward = self.remapReward(d)
        return reward
    
    def add_marker(self,goal=[]):
        self.marker_object_publisher = rospy.Publisher('/marker_basic',Marker,queue_size=1)
        self.rate = rospy.Rate(120)
        self.marker_object = Marker()
        self.marker_object.header.frame_id = "base_link"
        self.marker_object.header.stamp = rospy.get_rostime()
        self.marker_object.type = Marker.SPHERE
        self.marker_object.action = Marker.ADD

        my_point = geometry_msgs.msg.Point()
        my_point.x = goal[0]
        my_point.y = goal[1]
        my_point.z = goal[2]

        self.marker_object.pose.position = my_point

        self.marker_object.pose.orientation.x = 0.0
        self.marker_object.pose.orientation.y = 0.0
        self.marker_object.pose.orientation.z = 0.0
        self.marker_object.pose.orientation.w = 1.0

        self.marker_object.scale.x = 0.05
        self.marker_object.scale.y = 0.05
        self.marker_object.scale.z = 0.05

        self.marker_object.color.r = 0.0
        self.marker_object.color.g = 1.0
        self.marker_object.color.b = 0.0
        self.marker_object.color.a = 1.0

        self.marker_object.lifetime = rospy.Duration(0)

        self.marker_object_publisher.publish(self.marker_object)
