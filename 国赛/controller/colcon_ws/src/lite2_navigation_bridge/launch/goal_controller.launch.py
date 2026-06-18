from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument("pose_topic", default_value="/orbslam3/pose"),
            DeclareLaunchArgument("pose_type", default_value="pose_stamped"),
            DeclareLaunchArgument("receiver_ip", default_value="127.0.0.1"),
            DeclareLaunchArgument("receiver_port", default_value="5005"),
            DeclareLaunchArgument("target_x", default_value="1.0"),
            DeclareLaunchArgument("target_y", default_value="0.0"),
            DeclareLaunchArgument("target_yaw", default_value="0.0"),
            Node(
                package="lite2_navigation_bridge",
                executable="goal_controller",
                name="lite2_goal_controller",
                output="screen",
                parameters=[
                    {
                        "pose_topic": LaunchConfiguration("pose_topic"),
                        "pose_type": LaunchConfiguration("pose_type"),
                        "receiver_ip": LaunchConfiguration("receiver_ip"),
                        "receiver_port": LaunchConfiguration("receiver_port"),
                        "target_x": LaunchConfiguration("target_x"),
                        "target_y": LaunchConfiguration("target_y"),
                        "target_yaw": LaunchConfiguration("target_yaw"),
                    }
                ],
            ),
        ]
    )
