from setuptools import setup

package_name = "lite2_navigation_bridge"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml", "README.md"]),
        (f"share/{package_name}/launch", ["launch/goal_controller.launch.py"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="TheDog2",
    maintainer_email="user@example.com",
    description="Bridge ROS2 SLAM pose to Jueying Lite2 UDP motion commands.",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "goal_controller = lite2_navigation_bridge.goal_controller:main",
        ],
    },
)
