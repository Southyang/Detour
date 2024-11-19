#!/usr/bin/env python3
import math
import rospy
import tf
from geometry_msgs.msg import TransformStamped
from gazebo_msgs.srv import GetModelState, GetModelStateRequest
from std_msgs.msg import Float32

# 负责发布机器人TF变换和机器人到最近的障碍物或者其他机器人的距离
POSITION_LIST = []
MAX_DIST = 20.

def get_initial_state(robot_name):
    """从Gazebo的/model_state服务获取model的初始状态"""
    rospy.wait_for_service('gazebo/get_model_state')
    get_model_state = rospy.ServiceProxy('gazebo/get_model_state', GetModelState)
    request = GetModelStateRequest()
    request.model_name = robot_name
    response = get_model_state(request)

    POSITION_LIST.append((response.pose.position.x, response.pose.position.y))

    return response.pose


def publish_world_to_odom_transform(robot_name, initial_pose):
    br = tf.TransformBroadcaster()
    t = TransformStamped()

    t.header.stamp = rospy.Time.now()
    t.header.frame_id = "world"
    t.child_frame_id = robot_name + "/base_link"
    t.transform.translation.x = initial_pose.position.x
    t.transform.translation.y = initial_pose.position.y
    t.transform.translation.z = initial_pose.position.z
    t.transform.rotation.x = initial_pose.orientation.x
    t.transform.rotation.y = initial_pose.orientation.y
    t.transform.rotation.z = initial_pose.orientation.z
    t.transform.rotation.w = initial_pose.orientation.w

    br.sendTransformMessage(t)

if __name__ == '__main__':
    rospy.init_node('tf_transfer_world', anonymous=True)
    robot_num = rospy.get_param('~robot_num')
    stage_index = rospy.get_param('~stage_index')
    Publisher_list = []
    for i in range(robot_num):
        robot_name = 'robot'+str(i)
        dist_publisher = rospy.Publisher(robot_name + '/MinDist', Float32, queue_size=10)
        Publisher_list.append(dist_publisher)

    Obstacle_list = []
    if stage_index == 5:
        Obstacle_list = ["stone_50*50*30","stone_50*50*30_1","stone_50*50*30_2","stone_50*50*30_3","stone_50*50*30_4"]
    elif stage_index == 4:
        Obstacle_list = ["stone_50*50*30","stone_50*50*30_1","stone_50*50*30_2","stone_50*50*30_3","stone_50*50*30_4","stone_50*50*30_5"]
    elif stage_index == 3:
        Obstacle_list = []
    elif stage_index == 2:
        Obstacle_list = ["stone_50*50*30_0", "stone_50*50*30_1", "stone_50*50*30_2", "stone_50*50*30_3", "stone_50*50*30_4"]
    else:
        Obstacle_list = ["stone_50*50*30","stone_50*50*30_0","stone_50*50*30_1","stone_50*50*30_2","stone_50*50*30_3"]

    rate = rospy.Rate(10)  # HZ

    while not rospy.is_shutdown():
        try:
            POSITION_LIST.clear()
            for i in range(robot_num):
                robot_name = 'robot'+str(i)
                initial_pose = get_initial_state(robot_name)
                publish_world_to_odom_transform(robot_name, initial_pose)

            for i, name in enumerate(Obstacle_list):
                get_initial_state(name)

            dist_list = [MAX_DIST]*robot_num
            for i in range(robot_num):
                j=i+1
                while(j<len(POSITION_LIST)):
                    dist_temp = math.sqrt((POSITION_LIST[i][0]-POSITION_LIST[j][0])**2 + (POSITION_LIST[i][1]-POSITION_LIST[j][1])**2)
                    dist_list[i] = min(dist_list[i], dist_temp)
                    if j<robot_num:
                        dist_list[j] = min(dist_list[j], dist_temp)
                    j+=1

            for i in range(robot_num):
                Publisher_list[i].publish(dist_list[i])
        except:
            pass

        rate.sleep()
