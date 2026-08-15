from setuptools import setup
import os
from glob import glob

package_name = 'arm_grasp'

setup(
    name=package_name,
    version='1.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='team',
    maintainer_email='team@contest.cn',
    description='机械臂抓取任务模块',
    license='MIT',
    entry_points={
        'console_scripts': [
            'vision_node = arm_grasp.vision_node:main',
            'arm_control_node = arm_grasp.arm_control_node:main',
            'inspection_memory_node = arm_grasp.inspection_memory_node:main',
            'task_manager_node = arm_grasp.task_manager_node:main',
            'visualization_node = arm_grasp.visualization_node:main',
        ],
    },
)
