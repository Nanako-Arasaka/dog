from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'device', default_value='/dev/ttyUSB0',
            description='串口设备路径'),
        DeclareLaunchArgument(
            'baudrate', default_value='1000000',
            description='波特率 (Jetson↔STM32 = 1Mbps)'),

        Node(
            package='ros_robot_controller',
            executable='serial_bridge_node',
            name='serial_bridge_node',
            parameters=[{
                'device': LaunchConfiguration('device'),
                'baudrate': LaunchConfiguration('baudrate'),
            }],
            output='screen',
        ),
    ])
