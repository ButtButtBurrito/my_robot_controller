from setuptools import find_packages, setup

package_name = 'my_robot_controller'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='burrito',
    maintainer_email='burrito@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            "test_node = my_robot_controller.my_first_node:main",
            "draw_circle = my_robot_controller.draw_circle:main",
            "my_drawing = my_robot_controller.my_drawing:main",
            "pose_subscriber = my_robot_controller.pose_subscriber:main",
            "turtle_controller = my_robot_controller.turtle_controller:main",
            "piper_visualizer_controller = my_robot_controller.piper_visualizer_controller:main",
            "arm_hold = my_robot_controller.arm_hold:main",
            "move_it_driver = my_robot_controller.move_it_driver:main",
            "piper_hardware_controller = my_robot_controller.piper_hardware_controller:main",
            "camera_to_arm_bridge = my_robot_controller.camera_to_arm_bridge:main",
            "calibration_gui = my_robot_controller.calibration_gui:main",
            "aruco_sequence = my_robot_controller.aruco_sequence:main",
            "eye_in_hand_practice = my_robot_controller.eye_in_hand_practice:main",
            "eye_in_hand_v2 = my_robot_controller.eye_in_hand_v2:main",
            "pick_and_place = my_robot_controller.pick_and_place:main",
            "skill_server = my_robot_controller.skill_server:main",
            "scene_map = my_robot_controller.scene_map:main"
        
        ],
    },
)
