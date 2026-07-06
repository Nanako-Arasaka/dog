from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
import os
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    pkg_share = get_package_share_directory('arm_grasp')
    config_path = os.path.join(pkg_share, 'config', 'grasp_config.yaml')

    return LaunchDescription([
        DeclareLaunchArgument('config_path', default_value=config_path),
        DeclareLaunchArgument('auto_start_on_targets', default_value='true'),

        # 节点1: 视觉识别
        Node(
            package='arm_grasp',
            executable='vision_node',
            name='vision_node',
            output='screen',
            parameters=[{
                'config_path': LaunchConfiguration('config_path'),
                'target_color': 'red',
                'min_area': 500,
                'min_confidence': 0.1,
            }]
        ),

        # 节点2: 机械臂控制
        Node(
            package='arm_grasp',
            executable='arm_control_node',
            name='arm_control_node',
            output='screen',
            parameters=[{'config_path': LaunchConfiguration('config_path')}]
        ),

        # 节点3: 巡检记忆
        Node(
            package='arm_grasp',
            executable='inspection_memory_node',
            name='inspection_memory_node',
            output='screen'
        ),

        # 节点4: 任务管理
        Node(
            package='arm_grasp',
            executable='task_manager_node',
            name='task_manager_node',
            output='screen',
            parameters=[{
                'config_path': LaunchConfiguration('config_path'),
                'auto_start_on_targets': LaunchConfiguration('auto_start_on_targets'),
            }]
        ),

        # 节点5: 可视化
        Node(
            package='arm_grasp',
            executable='visualization_node',
            name='visualization_node',
            output='screen'
        ),
    ])
