from setuptools import setup
import os
from glob import glob

package_name = 'ros_robot_controller'

setup(
    name=package_name,
    version='1.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Hiwonder',
    maintainer_email='info@hiwonder.com',
    description='ROS2 serial bridge for Hiwonder STM32 robot controller (JetArm)',
    license='MIT',
    entry_points={
        'console_scripts': [
            'serial_bridge_node = ros_robot_controller.serial_bridge_node:main',
        ],
    },
)
