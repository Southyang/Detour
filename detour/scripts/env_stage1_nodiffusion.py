#! /usr/bin/env python3

import copy
import gc
import math
import time

import geometry_msgs
import nav_msgs
import numpy as np
import rospy
import tf
from gazebo_msgs.msg import ModelState,ModelStates,ContactsState
from gazebo_msgs.srv import SetModelState
from geometry_msgs.msg import Twist, TransformStamped, PoseWithCovarianceStamped, PoseStamped
from std_srvs.srv import Empty
from std_msgs.msg import Int32, Float32
from sensor_msgs.msg import CompressedImage, Imu, NavSatFix,PointCloud2,LaserScan
import sensor_msgs.point_cloud2 as pc2
from rosgraph_msgs.msg import Clock
from nav_msgs.msg import Odometry, Path
from visualization_msgs.msg import Marker,MarkerArray
import cv2
from cv_bridge import CvBridge

MAP_SIZE = 10.   # 2倍为地图长宽
MAX_RANGE = 10.  # 激光的最远距离
MAX_STEP = 750  # 视为超时
ARRIVE_DIST = 1.
TIME_DELTA = 0.05
IMPROVED_REWARD = True # If use improved reward function
IMAGE_SIZE = (80, 45)
MAX_PUNISH = 25.

COlOR_LIST = [[1,0,0],[0,1,0],[0,0,1],[1,0.5,0],[0.5,0.5,0.5],[0.5,0,0.5]]

class Environment:
    def __init__(self,beam_num, index, num_env):
        self.index = index
        self.num_env = num_env
        rospy.init_node('GazeboEnv_'+str(index), anonymous=None)
        self.nav_path_received = False # 是否接收到路径

        self.beam_num = beam_num
        self.scan = None
        self.scan_memory = None
        # used in reset_world
        self.self_speed = [0.0, 0.0]
        self.step_goal = [0., 0.]
        self.step_r_cnt = 0.
        self.tilt_angel = [0.0, 0.0]
        self.min_collision_dist = MAX_RANGE
        self.pre_min_collision_dist = MAX_PUNISH

        # used in generate goal point
        self.map_size = np.array([8., 8.], dtype=np.float32)  # 20x20m
        self.goal_size = ARRIVE_DIST    # 小于算到达

        self.robot_value = 10.
        self.goal_value = 0.

        self.init_pose = None
        self.is_collision = False

        # for get reward and terminate
        self.stop_counter = 0

        # 压缩图片解析
        self.bridge = CvBridge()
        # 启用
        self.subscribe()
        self.publish()
        self.service()

        # # Wait until the first callback
        self.speed = None
        self.state = None
        self.speed_GT = None
        self.state_GT = None
        while self.scan is None or self.speed is None or self.state is None\
                or self.speed_GT is None or self.state_GT is None:
            pass

        rospy.sleep(1.)
        # # What function to call when you ctrl + c
        # rospy.on_shutdown(self.shutdown)
    
    def next(self, action):
        self.take_action(action)

    def subscribe(self):
        # Subscribe to camera topic
        camera_topic = '/robot'+ str(self.index) +'/realsense/color/image_raw/compressed'
        self.camera_sub = rospy.Subscriber(camera_topic, CompressedImage, self.camera_callback)

        # imu_topic = '/robot'+ str(self.index) +'/imu/data'
        # self.imu_sub = rospy.Subscriber(imu_topic,Imu, self.imu_callback)

        # gps_topic = '/robot'+ str(self.index) +'/navsat/fix'
        # self.gps_sub = rospy.Subscriber(gps_topic,NavSatFix,self.gps_callback)

        # odom_topic = '/robot' + str(self.index) + '/husky_velocity_controller/odom'
        # self.odom_sub = rospy.Subscriber(odom_topic, Odometry, self.odometry_callback)

        self.model_state_sub = rospy.Subscriber('/gazebo/model_states', ModelStates, self.model_state_callback)

        dist_topic = '/robot'+ str(self.index) +'/MinDist'
        self.dist_sub = rospy.Subscriber(dist_topic, Float32, self.dist_callback)

        # laser_topic = '/robot'+ str(self.index) +'/points'
        # self.laser_sub = rospy.Subscriber(laser_topic, PointCloud2, self.laser_scan_callback)

        laser_2D_topic = '/robot'+ str(self.index) +'/front/scan'
        self.laser_2D_sub = rospy.Subscriber(laser_2D_topic, LaserScan, self.laser_2D_scan_callback)

        bumper_topic = '/robot'+ str(self.index) +'/bumper'
        self.bumper_sub = rospy.Subscriber(bumper_topic, ContactsState,self.bumper_callback)

        self.sim_clock = rospy.Subscriber('clock', Clock, self.sim_clock_callback)

        # 订阅a star规划的路径
        nav_path_topic = '/robot' + str(self.index) + '/nav_path'
        self.nav_path = rospy.Subscriber(nav_path_topic, Path, self.nav_path_callback)

    def publish(self):
        # Publisher
        # 发布速度控制
        cmd_vel_topic = '/robot' + str(self.index) + '/husky_velocity_controller/cmd_vel'
        self.cmd_vel = rospy.Publisher(cmd_vel_topic, Twist, queue_size=10)
        
        # 发布目标点标记
        goal_marker_topic = '/robot' + str(self.index) + '/goal_point'
        self.goal_marker_pub = rospy.Publisher(goal_marker_topic, MarkerArray, queue_size=3)

        # 发布清除路径缓存
        clear_path_buffer_topic = '/path/buffer/clear'
        self.clear_path_buffer = rospy.Publisher(clear_path_buffer_topic, Int32, queue_size=10)

        # 发布起点
        init_pose_topic = '/robot' + str(self.index) + '/initialpose'
        self.init_pose_pub = rospy.Publisher(init_pose_topic, PoseWithCovarianceStamped, queue_size=10)

        # 发布终点
        goal_pose_topic = '/robot' + str(self.index) + '/move_base_simple/goal'
        self.goal_pose_pub = rospy.Publisher(goal_pose_topic, PoseStamped, queue_size=10)

    def nav_path_callback(self, msg):
        # 计算角度，A star算法得到的第十个点与初始点的角度，当前点和初始点的角度，如果小于0.2弧度，认为对齐
        # 需要知道a star算法的步长，以及当前算法的步长
        try:
            # if len(msg.poses) <= 10:
            #     self.astar_pos = [(pose.pose.position.x, pose.pose.position.y) for pose in msg.poses]
            # else:
            #     step = max(1, min(len(msg.poses), 100) // 10)
            #     self.astar_pos = [(msg.poses[i].pose.position.x, msg.poses[i].pose.position.y) for i in
            #                       range(0, min(len(msg.poses), 100), step)]
            # rospy.loginfo(f"nav_path to be received: {msg}")
            if len(msg.poses) == 0:
                self.astar_pos = []
            elif len(msg.poses) <= 10:
                self.astar_pos = (msg.poses[len(msg.poses) - 1].pose.position.x, msg.poses[len(msg.poses) - 1].pose.position.y)
            else:
                self.astar_pos = (msg.poses[10].pose.position.x, msg.poses[10].pose.position.y)
            self.nav_path_received = True

        except Exception as e:
            print(e)

        finally:
            gc.collect()  # 强制垃圾回收


    def pub_init_pose(self, x, y):
        # 机器人坐标 x, y

        initial_pose = PoseWithCovarianceStamped()
        initial_pose.header.stamp = rospy.Time.now()
        initial_pose.header.frame_id = "map"

        initial_pose.pose.pose.position.x = x
        initial_pose.pose.pose.position.y = y
        initial_pose.pose.pose.position.z = 0.0

        initial_pose.pose.pose.orientation.x = 0.0
        initial_pose.pose.pose.orientation.y = 0.0
        initial_pose.pose.pose.orientation.z = 0.0
        initial_pose.pose.pose.orientation.w = 1.0

        # 设置协方差矩阵
        initial_pose.pose.covariance = [0.0] * 36
        initial_pose.pose.covariance[0] = 0.25
        initial_pose.pose.covariance[-1] = 0.06853891945200942

        self.init_pose_pub.publish(initial_pose)
        # rospy.loginfo(f"pub_init_pose, {self.index}")

    def pub_goal_pose(self, x, y):
        # 目标点坐标 x, y

        goal_pose = PoseStamped()
        goal_pose.header.stamp = rospy.Time.now()
        goal_pose.header.frame_id = "map"

        goal_pose.pose.position.x = x
        goal_pose.pose.position.y = y
        goal_pose.pose.position.z = 0.0

        goal_pose.pose.orientation.x = 0.0
        goal_pose.pose.orientation.y = 0.0
        goal_pose.pose.orientation.z = 0.0
        goal_pose.pose.orientation.w = 1.0

        self.goal_pose_pub.publish(goal_pose)
        # rospy.loginfo(f"pub_goal_pose, {self.index}")

    def pub_pose(self, x, y):
        self.pub_init_pose(x, y) # 发布起点
        self.pub_goal_pose(self.goal_point[0], self.goal_point[1]) # 发布终点


    def if_align(self, last_pos, now_pos):
        # 检查是否接收到nav_path
        start_wait_time = time.time()
        while not self.nav_path_received:
            elapsed_time = time.time() - start_wait_time
            if elapsed_time > 3:
                rospy.loginfo(f"Waiting for nav_path to be received, {elapsed_time:.2f} seconds, robot {self.index}")
                self.pub_pose(last_pos[0], last_pos[1])
                rospy.loginfo(f"pub_pose, robot {self.index}")
            rospy.sleep(0.1)  # 等待0.1秒
        self.nav_path_received = False

        total_wait_time = time.time() - start_wait_time
        if total_wait_time > 5:
            rospy.loginfo(f"nav_path received. Total wait time: {total_wait_time:.2f} seconds robot {self.index}")

        if len(self.astar_pos) == 0:
            return False

        # 计算上一个位置和当前位置之间的方向向量
        direction_last_to_now = np.arctan2(now_pos[1] - last_pos[1], now_pos[0] - last_pos[0])
        direction_last_to_astar = np.arctan2(self.astar_pos[1] - last_pos[1], self.astar_pos[0] - last_pos[0])
        direction_diff = abs(direction_last_to_now - direction_last_to_astar)
        # 判断方向是否一致
        if direction_diff <= 0.5: # 0.2弧度约11.5度
            # rospy.loginfo(f"Direction from {last_pos} to {now_pos} is aligned with the path")
            return True
        return False

        # 计算当前位置与nav_path各点的距离
        # if len(self.astar_pos) == 0:
        #     return False
        # distances = [np.sqrt((now_pos[0] - x) ** 2 + (now_pos[1] - y) ** 2) for x, y in self.astar_pos]
        # min_distance = min(distances)
        #
        # # 判断当前位置是否接近nav_path上的某个点
        # if min_distance <= 0.2:
        #     # rospy.loginfo(f"Position {now_pos} is close to the path with a distance of {min_distance:.2f}")
        #     return True
        #
        # # 计算上一个位置和当前导航路径点之间的方向向量
        # direction_last_to_now = np.arctan2(now_pos[1] - last_pos[1], now_pos[0] - last_pos[0])
        # for (x, y) in self.astar_pos:
        #     direction_last_to_astar = np.arctan2(y - last_pos[1], x - last_pos[0])
        #     direction_diff = abs(direction_last_to_now - direction_last_to_astar)
        #     # 判断方向是否一致
        #     if direction_diff <= 0.2: # 0.2弧度约11.5度
        #         # rospy.loginfo(f"Direction from {last_pos} to {now_pos} is aligned with the path")
        #         return True
        # return False

    def service(self):
        # Service
        self.reset_sim = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)
        self.set_model_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
        self.unpause = rospy.ServiceProxy("/gazebo/unpause_physics", Empty)
        self.pause = rospy.ServiceProxy("/gazebo/pause_physics", Empty)
        self.reset_proxy = rospy.ServiceProxy("/gazebo/reset_world", Empty)

    def camera_callback(self, msg):
        try:
            # if compressed image to show
            # img = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding="bgr8")
            # img_resized = cv2.resize(img, (160, 90))
            # cv2.imshow("robot"+str(self.index), img_resized)
            # cv2.waitKey(1)
            self.image = msg.data

        except Exception as e:
            print(e)

    def imu_callback(self,data):
        pass

    def gps_callback(self,data):
        pass

    def odometry_callback(self,odometry):
        Quaternions = odometry.pose.pose.orientation
        Euler = tf.transformations.euler_from_quaternion([Quaternions.x, Quaternions.y, Quaternions.z, Quaternions.w])
        self.state = [odometry.pose.pose.position.x, odometry.pose.pose.position.y, Euler[2]]
        self.speed = [odometry.twist.twist.linear.x, odometry.twist.twist.angular.z]
        self.state_GT = self.state
        self.speed_GT = self.speed

    def model_state_callback(self, msg):
        robot_name = 'robot'+str(self.index)
        index = msg.name.index(robot_name)

        position = msg.pose[index].position
        orientation = msg.pose[index].orientation
        linear = msg.twist[index].linear
        angular = msg.twist[index].angular

        Euler = tf.transformations.euler_from_quaternion([orientation.x, orientation.y, orientation.z, orientation.w])
        self.state = [position.x, position.y, Euler[2]]
        self.state_GT = self.state
        # 倾斜角度（横滚角和俯仰角）
        self.tilt_angel = [Euler[0], Euler[1]]
        self.speed = [linear.x, angular.z]
        self.speed_GT = self.speed

    def dist_callback(self, msg):
        dist = msg.data
        # if dist < 1:
        #     self.is_collision = True

        self.min_collision_dist = dist

    def get_self_speed(self):
        return self.speed

    def get_self_state(self):
        return self.state

    def get_self_stateGT(self):
        return self.state_GT

    def get_self_speedGT(self):
        return self.speed_GT

    def laser_scan_callback(self,data):
        # 从ROS消息中提取点云数据
        self.scan = list(pc2.read_points(data, skip_nans=True, field_names=("x", "y", "z")))

    def laser_2D_scan_callback(self,data):
        # 从ROS消息中提取点云数据
        self.scan = np.array(data.ranges)

    def bumper_callback(self,data):
        if data.states:
            self.is_collision = True

    def sim_clock_callback(self, clock):
        self.sim_time = clock.clock.secs + clock.clock.nsecs / 1000000000.

    def reset_simulation(self):
        # Call the reset simulation service
        self.reset_sim()

    def get_sim_time(self):
        return self.sim_time

    def reset_world(self):
        # rospy.wait_for_service("/gazebo/reset_world")
        # self.reset_proxy()

        self.self_speed = [0.0, 0.0]
        self.step_goal = [0., 0.]
        self.step_r_cnt = 0.
        self.start_time = time.time()
        self.is_collision = False

    def generate_random_pose(self):
        x = np.random.uniform(-MAP_SIZE, MAP_SIZE)
        y = np.random.uniform(-MAP_SIZE, MAP_SIZE)
        dis = np.sqrt(x ** 2 + y ** 2)
        while (dis > MAP_SIZE) and not rospy.is_shutdown():
            x = np.random.uniform(-MAP_SIZE, MAP_SIZE)
            y = np.random.uniform(-MAP_SIZE, MAP_SIZE)
            dis = np.sqrt(x ** 2 + y ** 2)
        theta = np.random.uniform(0, 2 * np.pi)
        return [x, y, theta]
    
    def clear_path_buffer_pub(self):
        self.clear_path_buffer.publish(self.index)
    
    def reset_pose(self):
        random_pose = self.generate_random_pose()
        rospy.sleep(0.01)
        self.control_pose(random_pose)
        self.clear_path_buffer_pub()
        time.sleep(TIME_DELTA)
        self.is_collision = False
        self.min_collision_dist = MAX_RANGE
        self.pre_min_collision_dist = MAX_RANGE

    def control_pose(self, pose):
        model_state = ModelState()
        model_state.model_name = 'robot'+str(self.index)
        assert len(pose)==3
        model_state.pose.position.x = pose[0]
        model_state.pose.position.y = pose[1]
        model_state.pose.position.z = 0.2

        quaternion = tf.transformations.quaternion_from_euler(0,0,pose[2])
        model_state.pose.orientation.x = quaternion[0]
        model_state.pose.orientation.y = quaternion[1]
        model_state.pose.orientation.z = quaternion[2]
        model_state.pose.orientation.w = quaternion[3]

        self.set_model_state(model_state)

    def control_vel(self, action):
        move_cmd = Twist()
        move_cmd.linear.x = action[0]
        move_cmd.linear.y = 0.
        move_cmd.linear.z = 0.
        move_cmd.angular.x = 0.
        move_cmd.angular.y = 0.
        move_cmd.angular.z = action[1]
        self.cmd_vel.publish(move_cmd)

    def generate_goal_point(self):
        [x_g, y_g] = self.generate_random_goal()
        self.goal_point = [x_g, y_g]
        [x, y] = self.get_local_goal()

        self.pre_distance = np.sqrt(x ** 2 + y ** 2)
        self.distance = copy.deepcopy(self.pre_distance)

        self.publish_goal_markers(self.goal_point)

    def generate_random_goal(self):
        self.init_pose = self.get_self_stateGT()
        x = np.random.uniform(-MAP_SIZE, MAP_SIZE)
        y = np.random.uniform(-MAP_SIZE, MAP_SIZE)
        dis_origin = np.sqrt(x ** 2 + y ** 2)   # 距离原点
        dis_goal = np.sqrt((x - self.init_pose[0]) ** 2 + (y - self.init_pose[1]) ** 2) # 距离小车
        while (dis_origin > MAP_SIZE or dis_goal > MAP_SIZE or dis_goal < MAP_SIZE*2/3) and not rospy.is_shutdown():
            x = np.random.uniform(-MAP_SIZE, MAP_SIZE)
            y = np.random.uniform(-MAP_SIZE, MAP_SIZE)
            dis_origin = np.sqrt(x ** 2 + y ** 2)
            dis_goal = np.sqrt((x - self.init_pose[0]) ** 2 + (y - self.init_pose[1]) ** 2)

        return [x, y]
    
    def get_local_goal(self):
        [x, y, theta] = self.get_self_stateGT()
        [goal_x, goal_y] = self.goal_point
        local_x = (goal_x - x) * np.cos(theta) + (goal_y - y) * np.sin(theta)
        local_y = -(goal_x - x) * np.sin(theta) + (goal_y - y) * np.cos(theta)

        self.orientation_to_goal = math.atan2(local_y, local_x)
        self.orientation = theta

        return [local_x, local_y]
    
    def get_local_orientation(self):

        return [self.orientation, self.orientation_to_goal, self.tilt_angel[0], self.tilt_angel[1]]
    
    def publish_goal_markers(self, goal):
        # Publish visual data in Rviz
        markerArray = MarkerArray()
        marker = Marker()
        marker.header.frame_id = "world"
        marker.type = marker.CYLINDER
        marker.action = marker.ADD
        marker.scale.x = 0.5
        marker.scale.y = 0.5
        marker.scale.z = 0.01
        marker.color.a = 1.0
        marker.color.r = COlOR_LIST[self.index][0]
        marker.color.g = COlOR_LIST[self.index][1]
        marker.color.b = COlOR_LIST[self.index][2]
        marker.pose.orientation.w = 1.0
        marker.pose.position.x = goal[0]
        marker.pose.position.y = goal[1]
        marker.pose.position.z = 0

        markerArray.markers.append(marker)

        self.goal_marker_pub.publish(markerArray)
        
    def get_laser_observation(self):
        # 3D
        # if len(self.scan) == 0:
        #     return self.scan_memory

        # data = self.scan
        # data = np.array(data)
        # step = len(data) / float(self.beam_num)
        # indices = np.arange(0, len(data), step).astype(int)
        # selected_points = data[indices]

        # total_beam = selected_points/MAX_RANGE
        # total_beam = total_beam.transpose()
        # self.scan_memory = total_beam
        # return total_beam

        # 2D
        if len(self.scan) == 0:
            return self.scan_memory
        
        scan = self.scan
        scan[np.isnan(scan)] = MAX_RANGE
        scan[np.isinf(scan)] = MAX_RANGE
        raw_beam_num = len(scan)
        sparse_beam_num = self.beam_num
        step = float(raw_beam_num) / sparse_beam_num
        sparse_scan_left = []
        index = 0.
        for _ in range(int(sparse_beam_num / 2)):
            sparse_scan_left.append(scan[int(index)])
            index += step
        sparse_scan_right = []
        index = raw_beam_num - 1.
        for _ in range(int(sparse_beam_num / 2)):
            sparse_scan_right.append(scan[int(index)])
            index -= step
        scan_sparse = np.concatenate((sparse_scan_left, sparse_scan_right[::-1]), axis=0)

        total_beam = scan_sparse / MAX_RANGE - 0.5
        self.scan_memory = total_beam

        return total_beam
    
    def get_camera_observation(self):
        compressed_image = np.frombuffer(self.image, np.uint8)
        image_obs = cv2.imdecode(compressed_image, cv2.IMREAD_COLOR)

        # cv2.namedWindow('Image', cv2.WINDOW_NORMAL)
        # cv2.resizeWindow('Image', 1500, 1500)
        # cv2.imshow('Image', image_obs)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # print(f"Initial img shape: {image_obs.shape}")

        image_obs = cv2.resize(image_obs, IMAGE_SIZE)

        # cv2.namedWindow('Image', cv2.WINDOW_NORMAL)
        # cv2.resizeWindow('Image', 1500, 1500)
        # cv2.imshow('Image', image_obs)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # print(f"result img shape: {image_obs.shape}")

        image_obs = image_obs.astype(np.float32) / 255.0

        return image_obs

    def get_crash_state(self):
        return self.is_collision
    
    def get_tilt_penalty(self, angle, angle_type):
        terminal = False
        angle = np.degrees(angle)
        # 根据角度类型设定阈值
        if angle_type == 'pitch':  # 俯仰角
            no_penalty_threshold = 35
            linear_penalty_threshold = 45
        elif angle_type == 'roll':  # 横滚角
            no_penalty_threshold = 30
            linear_penalty_threshold = 40
        else:
            raise ValueError('Invalid angle type')

        # 计算惩罚
        if angle <= no_penalty_threshold:
            penalty = 0
        elif angle <= linear_penalty_threshold:
            # 线性惩罚
            penalty = (angle - no_penalty_threshold) * (MAX_PUNISH / (linear_penalty_threshold - no_penalty_threshold))
        else:
            terminal = True
            penalty = MAX_PUNISH

        return penalty, terminal

    # ours stage1 reward
    def get_reward_and_terminate1(self, t):
        # 阶段一奖励函数
        reward_g = 0 # 靠近目标奖励
        reward_c = 0 # 碰撞惩罚
        reward_w = 0 # 角速度惩罚
        reward_t = 0 # 倾斜惩罚
        reward_s = 0 # 时间步惩罚
        result = 0
        terminate = False

        # 靠近目标奖励
        laser_scan = self.get_laser_observation()
        [x, y, theta] = self.get_self_stateGT()
        [v, w] = self.get_self_speedGT()
        self.pre_distance = copy.deepcopy(self.distance)
        self.distance = np.sqrt((self.goal_point[0] - x) ** 2 + (self.goal_point[1] - y) ** 2)
        if self.distance <= self.goal_size:
            terminate = True
            reward_g = MAX_PUNISH
            result = 'Reach Goal'
        else:
            reward_g = (self.pre_distance - self.distance) * 2.5

        # 碰撞惩罚
        is_crash = self.get_crash_state()
        if is_crash:
            terminate = True
            reward_c = MAX_PUNISH
            result = 'Crashed'
        elif self.min_collision_dist <= 2.5 and self.pre_min_collision_dist != MAX_RANGE:
            reward_c = (self.pre_min_collision_dist - self.min_collision_dist) * 2
        self.pre_min_collision_dist = self.min_collision_dist

        # 倾斜惩罚
        r_t1, terminate1 = self.get_tilt_penalty(abs(self.tilt_angel[0]), "roll")
        r_t2, terminate2 = self.get_tilt_penalty(abs(self.tilt_angel[1]), "pitch")
        if not terminate1 and not terminate2:
            reward_t = 0
        else:
            reward_t = MAX_PUNISH
            terminate = True
            result = 'Rollover'

        # 角速度惩罚
        if abs(w) > 0.7:
            reward_w = 0.1 * abs(w)

        # 时间步惩罚
        if t > MAX_STEP:
            terminate = True
            result = 'Time out'
            reward_s = MAX_PUNISH
        else:
            reward_s = 0.02

        if IMPROVED_REWARD:
            reward = reward_g - reward_c - reward_t - reward_s
        else:
            reward = reward_g - reward_c - reward_w

        if reward < 0:
            reward = max(reward, -MAX_PUNISH * 3)

        return reward, terminate, result