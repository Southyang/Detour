#!/usr/bin/env python3
import rospy
from geometry_msgs.msg import PoseStamped
from gazebo_msgs.srv import GetModelState, GetModelStateRequest
from nav_msgs.msg import Path
from std_msgs.msg import Int32

Path_list = []
Publisher_list = []
MAX_POSES = 30  # 缓存的路径最大数

def get_robot_state(robot_name):
    """从Gazebo的/model_state服务获取机器人的初始状态"""
    rospy.wait_for_service('gazebo/get_model_state')
    
    get_model_state = rospy.ServiceProxy('gazebo/get_model_state', GetModelState)
    request = GetModelStateRequest()
    request.model_name = robot_name
    response = get_model_state(request)
    return response.pose


def publish_path(robot_index, robot_pose):
    pose = PoseStamped()
    pose.header.stamp = rospy.Time.now()
    pose.header.frame_id = 'world'
    pose.pose.position.x = robot_pose.position.x
    pose.pose.position.y = robot_pose.position.y

    if len(Path_list[robot_index].poses) > MAX_POSES:
        Path_list[robot_index].poses.pop(0)

    Path_list[robot_index].header = pose.header
    Path_list[robot_index].poses.append(pose)
    Publisher_list[robot_index].publish(Path_list[robot_index])

def clear_path_buffer_callback(msg):
    Path_list[msg.data].poses = []


if __name__ == '__main__':
    rospy.init_node('path_pub_node', anonymous=True)
    robot_num = rospy.get_param('~robot_num')
    MAX_POSES = rospy.get_param('~max_poses')
    rate = rospy.Rate(1)  # HZ
    print("1111111111111111111111111111111")

    for i in range(robot_num):
        robot_name = 'robot'+str(i)
        path_publisher = rospy.Publisher(robot_name + '/path', Path, queue_size=10)
        Publisher_list.append(path_publisher)
        path = Path()
        Path_list.append(path)

    while not rospy.is_shutdown():
        try:
            rospy.Subscriber('/path/buffer/clear', Int32, clear_path_buffer_callback)
            for i in range(robot_num):
                robot_name = 'robot'+str(i)
                robot_pose = get_robot_state(robot_name)
                publish_path(i, robot_pose)
        except:
            pass

        rate.sleep()
