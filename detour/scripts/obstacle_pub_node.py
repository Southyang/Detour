#!/usr/bin/env python3
import rospy
from gazebo_msgs.srv import GetModelState, GetModelStateRequest
from visualization_msgs.msg import Marker,MarkerArray

Obstacle_list = []
# 发布目标点标记
Obstacle_marker_topic = '/obstacle/marker'
Obstacle_marker_pub = rospy.Publisher(Obstacle_marker_topic, MarkerArray, queue_size=10)

def get_obstacle_state(name):
    rospy.wait_for_service('gazebo/get_model_state')
    
    get_model_state = rospy.ServiceProxy('gazebo/get_model_state', GetModelState)
    request = GetModelStateRequest()
    request.model_name = name
    response = get_model_state(request)
    return response.pose



if __name__ == '__main__':
    rospy.init_node('obstacle_pub_node', anonymous=True)
    stage_index = rospy.get_param('~stage_index')
    
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


    rate = rospy.Rate(1)  # HZ

    while not rospy.is_shutdown():
        try:
            markerArray = MarkerArray()
            for i, name in enumerate(Obstacle_list):
                pose = get_obstacle_state(name)
                marker = Marker()
                marker.header.frame_id = "world"
                marker.type = marker.CUBE
                marker.action = marker.ADD
                marker.scale.x = 0.5
                marker.scale.y = 0.5
                marker.scale.z = 0.01
                marker.color.a = 1.0
                marker.color.r = 1
                marker.color.g = 1
                marker.color.b = 1
                marker.pose.orientation.w = 1.0
                marker.pose.position.x = pose.position.x
                marker.pose.position.y = pose.position.y
                marker.pose.position.z = 0
                marker.id = i

                markerArray.markers.append(marker)
                
            Obstacle_marker_pub.publish(markerArray)
        except:
            pass
        
        rate.sleep()
