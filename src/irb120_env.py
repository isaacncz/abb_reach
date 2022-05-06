import numpy as np
import rospy
# from gazebo_msgs.srv import GetWorldProperties, GetModelState
# from sensor_msgs.msg import JointState
# from openai_ros import robot_gazebo_env
import sys
import moveit_commander
import moveit_msgs.msg

from moveit_msgs.srv import GetPositionFK, GetPositionFKRequest
import geometry_msgs.msg
import copy
from std_msgs.msg import Header
from moveit_commander.conversions import pose_to_list
# import trajectory_msgs.msg
# from openai_ros.openai_ros_common import ROSLauncher
from sensor_msgs.msg import JointState

from math import pi
import os
from gym import spaces
import gym
# from gym import error
# from tf.transformations import euler_from_quaternion, quaternion_from_euler
# moveit reference
# http://docs.ros.org/en/jade/api/moveit_commander/html/classmoveit__commander_1_1move__group_1_1MoveGroupCommander.html#a6a4440597144e033cd1749e0e00716ca

# for sphere target
# from gazebo_msgs.msg import ModelState 
# from gazebo_msgs.srv import SetModelState

# distance
# from typing import Any, Dict, Union
from typing import Any, Dict, Optional, Tuple, Union
# generate random number
import random


from gym.envs.registration import register

reg = register(
            id="AbbReach-v0",
            entry_point='irb120_env:AbbEnv',
            max_episode_steps=10000,
        )


try:
    from math import pi, tau, dist, fabs, cos
except:  # For Python 2 compatibility
    from math import pi, fabs, cos, sqrt

    tau = 2.0 * pi

    def dist(p, q):
        return sqrt(sum((p_i - q_i) ** 2.0 for p_i, q_i in zip(p, q)))


def all_close(goal, actual, tolerance):
    """
    Convenience method for testing if the values in two lists are within a tolerance of each other.
    For Pose and PoseStamped inputs, the angle between the two quaternions is compared (the angle
    between the identical orientations q and -q is calculated correctly).
    @param: goal       A list of floats, a Pose or a PoseStamped
    @param: actual     A list of floats, a Pose or a PoseStamped
    @param: tolerance  A float
    @returns: bool
    """
    if type(goal) is list:
        for index in range(len(goal)):
            if abs(actual[index] - goal[index]) > tolerance:
                return False

    elif type(goal) is geometry_msgs.msg.PoseStamped:
        return all_close(goal.pose, actual.pose, tolerance)

    elif type(goal) is geometry_msgs.msg.Pose:
        x0, y0, z0, qx0, qy0, qz0, qw0 = pose_to_list(actual)
        x1, y1, z1, qx1, qy1, qz1, qw1 = pose_to_list(goal)
        # Euclidean distance
        d = dist((x1, y1, z1), (x0, y0, z0))
        # phi = angle between orientations
        cos_phi_half = fabs(qx0 * qx1 + qy0 * qy1 + qz0 * qz1 + qw0 * qw1)
        return d <= tolerance and cos_phi_half >= cos(tolerance / 2.0)

    return True



# class AbbEnv(robot_gazebo_env.RobotGazeboEnv):
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
        rospy.logdebug("Start abbenv INIT...")
        self.robot = moveit_commander.RobotCommander()
        self.scene = moveit_commander.PlanningSceneInterface()
        self.arm_group = moveit_commander.MoveGroupCommander("manipulator")
        self.arm_group.set_planner_id("RRTConnectkConfigDefault")
        # self.arm_group.set_planner_id("TRAC_IKKinematicsPlugin")
        rospy.logdebug("end var INIT...")

        # RL 
        # goal_range=0.3
        self.distance_threshold = 0.05
        self.reward_type = "dense"

        # self.goal_range_low = np.array([-goal_range / 2, -goal_range / 2, 0])
        # self.goal_range_high = np.array([goal_range / 2, goal_range / 2, goal_range])
        # print(self.goal_range_high )
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
        
        while (i<7):
            rqst.robot_state.joint_state.name.append('joint_'+str(i))
            rqst.robot_state.joint_state.position.append(self.arm_group.get_current_joint_values()[i-1])
            i+=1
        res = moveit_fk(rqst)
        # print(res.pose_stamped[0].pose.position)
        position = [0, 0, 0]
        position[0] = res.pose_stamped[0].pose.position.x
        position[1] = res.pose_stamped[0].pose.position.y
        position[2] = res.pose_stamped[0].pose.position.z
        position = np.array(position)
        self.achieved_goal = position
        # print('self.achieved_goal',self.achieved_goal)
        return position
        # rospy.logdebug("computed fk:", res)


    # def go_to_pose_goal(self, x,y,z):
    #     self.arm_group.set_goal_tolerance(0.05)
    #     pose_goal = geometry_msgs.msg.Pose()
    #     pose_goal.orientation.w = 1.000000

    #     # pose_goal.orientation.w = 0.519859
    #     pose_goal.position.x = x
    #     pose_goal.position.y = y
    #     pose_goal.position.z = z

    #     self.arm_group.set_pose_target(pose_goal)

    #     ## Now, we call the planner to compute the plan and execute it.
    #     plan = self.arm_group.go(wait=True)
    #     # Calling `stop()` ensures that there is no residual movement
    #     self.arm_group.stop()
    #     # It is always good to clear your targets after planning with poses.
    #     # Note: there is no equivalent function for clear_joint_value_targets()
    #     self.arm_group.clear_pose_targets()
    #     current_pose = self.arm_group.get_current_pose().pose
    #     return all_close(pose_goal, current_pose, 0.05)

    def plan_cartesian_path(self,x,y,z):

        waypoints = []
        scale = 1
        wpose = self.arm_group.get_current_pose().pose
        wpose.position.x = x
        wpose.position.y = y
        wpose.position.z = z

        waypoints.append(copy.deepcopy(wpose))

        # We want the Cartesian path to be interpolated at a resolution of 1 cm
        # which is why we will specify 0.01 as the eef_step in Cartesian
        # translation.  We will disable the jump threshold by setting it to 0.0,
        # ignoring the check for infeasible jumps in joint space, which is sufficient
        # for this tutorial.
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
        self.arm_group.set_goal_joint_tolerance(0.05)
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
        self.arm_group.set_goal_joint_tolerance(0.01)
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
        
        arrayOfPose = [    
            current_pose.position.x ,
            current_pose.position.y ,
            current_pose.position.z ,
            current_pose.orientation.x ,
            current_pose.orientation.y ,       
            current_pose.orientation.z ,
            current_pose.orientation.w ,
        ]
        currentPoseArray = np.array(arrayOfPose)
        # print("current pose:",currentPoseArray) #to normalize this value

        # joint_angles = self.arm_group.get_joint_value_target() # [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        # print("joint_angles: ",joint_angles)

        # return currentPoseArray.copy()

        return {
            "observation": currentPoseArray.copy(),
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
        rospy.logdebug("Waiting for /joint_states to be READY...")
        while self.joint_states is None and not rospy.is_shutdown():
            try:
                self.joint_states = self.robot.get_current_state()
                rospy.logdebug("Current /joint_states READY=>")
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
        self.move_joint_arm(0,0,0,0,0,0) #g o back to neutral position
        self.wait_time_for_execute_movement()
        # return self.desired_goal
        # print(self._get_obs())
        return self._get_obs()

    def _get_action_space(self)-> gym.spaces.Box:
        """Get environment action space.

        Returns:
            gym.spaces: Gym action space object.

        """
        return self.action_space 


    def step(self, action) -> Tuple[np.array, float, bool, dict]:
        if type(action) == list: action = np.array(action)

        joint_goal = self.arm_group.get_current_joint_values()

        self.arm_group.set_goal_joint_tolerance(0.01)

        joint_goal[0] = action[0] * 2.87979 * 0.1
        joint_goal[1] = action[1] * 1.91986 * 0.1
        joint_goal[2] = action[2] * 1.91986 * 0.1
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
        print("Position:",position)
        # print("compute reward here:",self.compute_reward(self.achieved_goal,self.desired_goal))
        reward, done = self.dense_reward(self.achieved_goal,self.desired_goal)
        info = {"is_success":self.is_success(self.achieved_goal,self.desired_goal)}
        # info = {"is_success": done}
        # info = {"is_success": self.task.is_success(obs["achieved_goal"], self.task.get_goal())}
        print("Reward:",reward," info:",info)
        # reward, done, info = self.reward(rs_state=rs_state, action=action)

        # if self.rs_state_to_info: info['rs_state'] = self.rs_state
        return state, reward, done, info

    # def reward(self, rs_state, action) -> Tuple[float, bool, dict]:
    #     done = False
    #     info = {}

    #     if self.elapsed_steps >= self.max_episode_steps:
    #         done = True
    #         info['final_status'] = 'max_steps_exceeded'
            
    #     return 0, done, info

    def _sample_goal(self) -> np.ndarray:
        """Randomize goal."""
        x = random.uniform(0.2, 0.4)
        y = random.uniform(-0.35, 0.35)
        z = random.uniform(0.2, 0.4)
        goal = np.array([x,y,z])

        # state_msg = ModelState()
        # state_msg.model_name = 'sphere50_no_collision'
        # state_msg.pose.position.x = x 
        # state_msg.pose.position.y = y
        # state_msg.pose.position.z = z
        # 
        # rospy.wait_for_service('/gazebo/set_model_state')
        # try:
        #     set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
        #     resp = set_state(state_msg)
        # except rospy.ServiceException as e:
        #     print ("Service call failed: %s" , e)
        # goal = self.np_random.uniform(self.goal_range_low, self.goal_range_high)
        print("sample goal (xyz):",goal)
        return goal

    def is_success(self, achieved_goal: np.ndarray, desired_goal: np.ndarray) -> Union[np.ndarray, float]:
        d = self.distance(achieved_goal, desired_goal)
        return 1-d 
        # return np.array(d < self.distance_threshold, dtype=np.float64)

    # def compute_reward(self, achieved_goal, desired_goal) -> Union[np.ndarray, float]:
    #     d = self.distance(achieved_goal, desired_goal)
    #     if self.reward_type == "sparse":
    #         return -np.array(d > self.distance_threshold, dtype=np.float64)
    #     else:
    #         return -d

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
        scale = 10

        if np.array_equal(desired_goal, achieved_goal):
            reward = 1 * scale
            done = True
        else:
            reward = -np.linalg.norm(desired_goal - achieved_goal) * scale
            done = False

        return reward, done

