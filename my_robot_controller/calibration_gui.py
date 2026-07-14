#!/usr/bin/env python3
import sys
import os
import subprocess
import signal
import yaml
import threading
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from ament_index_python.packages import get_package_share_directory
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QGroupBox, 
                             QLineEdit, QTextEdit, QCheckBox, QFileDialog, QComboBox)
from PyQt5.QtCore import QTimer, pyqtSignal, QThread, Qt

from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from sensor_msgs.msg import JointState
from builtin_interfaces.msg import Duration
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import Constraints, JointConstraint, PositionConstraint, OrientationConstraint, MoveItErrorCodes

class LaunchManager(QThread):
    log_signal = pyqtSignal(str)
    process_finished_signal = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.processes = {}

    def launch(self, name, command):
        if name in self.processes:
            self.log_signal.emit(f"Process {name} is already running.")
            return

        self.log_signal.emit(f"Launching {name}: {command}")
        
        # Start the process in a separate thread to avoid blocking GUI
        thread = QThread()
        thread.run = lambda: self._run_process(name, command)
        thread.start()
        # Note: In a real robust app, we'd manage threads better, but this works for simple GUI.
        # Actually, let's strictly use subprocess here and monitor it.
        # Reverting to direct subprocess call inside a thread is tricky. 
        # Better: Just Popen it and have a timer check it? Or a dedicated thread per process?
        # Let's use the pattern from launch_gui.py (which uses a thread per process implicitly via method call? No, it blocked?)
        # The previous launch_gui.py used a worker thread? No, it used `subprocess.Popen` inside `_run_process` which BLOCKS?
        # Wait, `launch_gui.py` implementation I viewed earlier:
        # `thread.run = lambda: self._run_process(name, command)` -> execution starts immediately on THAT thread object? 
        # Standard QThread usage is `moveToThread` or subclass `run`. Assigning `run` lambda is a bit hacky but works if `start()` calls it.
        # Let's stick to simple Popen and a timer to check status? No, blocking read is better for logs.
        # I'll use a subclass of QThread for each process.

    def _run_process(self, name, command):
        # This blocks until process ends
        process = subprocess.Popen(
            command, 
            shell=True, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT, 
            preexec_fn=os.setsid,
            text=True
        )
        self.processes[name] = process
        
        while True:
            line = process.stdout.readline()
            if not line and process.poll() is not None:
                break
            if line:
                self.log_signal.emit(f"[{name}] {line.strip()}")
        
        if name in self.processes:
            del self.processes[name]
        self.process_finished_signal.emit(name)
        self.log_signal.emit(f"Process {name} finished.")

    def start_process_thread(self, name, command):
        # We need to keep a reference to threads to prevent GC
        if not hasattr(self, 'threads'): self.threads = {}
        thread = QThread()
        self.threads[name] = thread
        # This is getting complicated to implement correctly in one go.
        # Let's use a simpler approach: Just Popen, and read stdout in a QTimer?
        # Or just run it.
        pass 
        # Actually, let's reuse the logic from launch_gui.py exactly: it defines `_run_process` and calls it in a Thread.
        import threading
        t = threading.Thread(target=self._run_process, args=(name, command))
        t.daemon = True
        t.start()

    def stop(self, name):
        if name in self.processes:
            self.log_signal.emit(f"Stopping {name}...")
            try:
                os.killpg(os.getpgid(self.processes[name].pid), signal.SIGTERM)
            except Exception as e:
                self.log_signal.emit(f"Error stopping {name}: {e}")
            # process dict cleanup happens in the thread loop
        else:
            self.log_signal.emit(f"{name} is not running.")

class CalibrationNode(Node):
    UR10_JOINTS = [
        "joint1", "joint2", "joint3",
        "joint4", "joint5", "joint6"
    ]

    def __init__(self):
        super().__init__('calibration_gui_node')
        self.joint_subscription = self.create_subscription(
            JointState,
            '/feedback/joint_states',
            self.joint_callback,
            10)
        self.current_joints = None
        
        self._action_client = None
        self._move_group_client = ActionClient(self, MoveGroup, '/move_action')
        
    def update_controller(self, topic):
        if self._action_client:
            self._action_client.destroy()
        self._action_client = ActionClient(self, FollowJointTrajectory, topic)
        self.get_logger().info(f"Updated Action Client to: {topic}")
        
    def joint_callback(self, msg):
        # Check if it's the right joints (ur10 standard)
        # For simplicity, we just store the message
        self.current_joints = msg

    def send_goal(self, idx, point_msg):
        if not self._action_client.wait_for_server(timeout_sec=1.0):
            return False
            
        goal_msg = FollowJointTrajectory.Goal()
        goal_msg.trajectory = point_msg
        
        self._action_client.send_goal_async(goal_msg)
        return True

    def send_moveit_goal(self, target_joints, plan_only=False):
        if not self._move_group_client.wait_for_server(timeout_sec=1.0):
            self.get_logger().error("MoveGroup action server not available!")
            return False

        goal = MoveGroup.Goal()
        goal.request.group_name = "arm" # Piper SRDF group name
        goal.request.allowed_planning_time = 5.0
        goal.request.num_planning_attempts = 10
        goal.request.max_velocity_scaling_factor = 0.1
        goal.request.max_acceleration_scaling_factor = 0.1
        
        # Create Joint Constraints
        constraints = Constraints()
        joint_names = [
            'joint1', 'joint2', 'joint3',
            'joint4', 'joint5', 'joint6'
        ]
        
        for i, name in enumerate(joint_names):
            jc = JointConstraint()
            jc.joint_name = name
            jc.position = target_joints[i]
            jc.tolerance_above = 0.001
            jc.tolerance_below = 0.001
            jc.weight = 1.0
            constraints.joint_constraints.append(jc)
            
        goal.request.goal_constraints.append(constraints)
        goal.planning_options.plan_only = plan_only 
        goal.planning_options.look_around = False
        goal.planning_options.replan = False

        self.get_logger().info(f"Sending MoveGroup goal (Plan Only: {plan_only})...")
        return self._move_group_client.send_goal_async(goal)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Piper ArUco Hand-Eye Calibration")
        self.resize(1000, 700)
        
        self.launch_manager = LaunchManager()
        self.launch_manager.log_signal.connect(self.log_message)
        self.launch_manager.process_finished_signal.connect(self.on_process_finished)
        
        # Ros Node in background thread
        rclpy.init()
        self.ros_node = CalibrationNode()
        # Initial controller
        self.ros_node.update_controller('/arm_controller/follow_joint_trajectory')
        
        import threading
        self.ros_thread = threading.Thread(target=rclpy.spin, args=(self.ros_node,), daemon=True)
        self.ros_thread.start()
        
        self.running_states = {
            "robot": False, "camera": False, "moveit": False,
            "aruco": False, "handeye": False
        }
        
        self.poses = [] # List of joint states
        self.current_pose_idx = -1
        # Save to SOURCE directory to ensure persistence across builds
        self.yaml_path = os.path.expanduser("~/agx_arm_ws/src/ur10_calibration/config/calibration_poses.yaml")
        
        # Create config dir if not exists (in case of fresh checkout)
        os.makedirs(os.path.dirname(self.yaml_path), exist_ok=True)

        self.init_ui()
        self.load_poses()

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QHBoxLayout()
        
        # Left Panel: Controls
        left_layout = QVBoxLayout()
        
        # 1. Hardware Launch
        hw_group = QGroupBox("1. Hardware")
        hw_layout = QVBoxLayout()
        
        # Robot
        robot_layout = QHBoxLayout()
        self.robot_fake_chk = QCheckBox("Fake Robot")
        
        self.robot_ip = QLineEdit("192.168.11.100")
        self.robot_ip.setPlaceholderText("Robot IP")
        self.robot_ip.setFixedWidth(120)
        
        self.robot_btn = QPushButton("Start Robot")
        self.robot_btn.clicked.connect(self.toggle_robot)
        self.robot_btn.setStyleSheet("background-color: #4CAF50; color: white;")
        self.robot_status = QLabel("INACTIVE")
        self.robot_status.setStyleSheet("color: gray; font-weight: bold;")
        
        robot_layout.addWidget(self.robot_fake_chk)
        robot_layout.addWidget(QLabel("IP:"))
        robot_layout.addWidget(self.robot_ip)
        robot_layout.addWidget(self.robot_btn)
        robot_layout.addWidget(self.robot_status)
        hw_layout.addLayout(robot_layout)
        
        # Camera
        cam_layout = QHBoxLayout()
        self.cam_fake_chk = QCheckBox("Fake Camera")
        self.cam_btn = QPushButton("Start Camera")
        self.cam_btn.clicked.connect(self.toggle_camera)
        self.cam_btn.setStyleSheet("background-color: #2196F3; color: white;")
        self.cam_status = QLabel("INACTIVE")
        self.cam_status.setStyleSheet("color: gray; font-weight: bold;")
        cam_layout.addWidget(self.cam_fake_chk)
        cam_layout.addWidget(self.cam_btn)
        cam_layout.addWidget(self.cam_status)
        hw_layout.addLayout(cam_layout)
        
        hw_group.setLayout(hw_layout)
        left_layout.addWidget(hw_group)
        
        # 2. Tools Launch
        tools_group = QGroupBox("2. Tools")
        tools_layout = QVBoxLayout()
        
        # MoveIt
        moveit_layout = QHBoxLayout()
        self.moveit_fake_chk = QCheckBox("Fake Hardware")
        self.moveit_btn = QPushButton("Start MoveIt/RViz")
        self.moveit_btn.clicked.connect(self.toggle_moveit)
        self.moveit_status = QLabel("INACTIVE")
        self.moveit_status.setStyleSheet("color: gray; font-weight: bold;")
        moveit_layout.addWidget(self.moveit_fake_chk)
        moveit_layout.addWidget(self.moveit_btn)
        moveit_layout.addWidget(self.moveit_status)
        tools_layout.addLayout(moveit_layout)
        
        # ArUco
        aruco_layout = QHBoxLayout()
        self.aruco_btn = QPushButton("Start ArUco Node")
        self.aruco_btn.clicked.connect(self.toggle_aruco)
        self.aruco_status = QLabel("INACTIVE")
        self.aruco_status.setStyleSheet("color: gray; font-weight: bold;")
        aruco_layout.addWidget(self.aruco_btn)
        aruco_layout.addWidget(self.aruco_status)
        tools_layout.addLayout(aruco_layout)
        
        # EasyHandEye
        he_layout = QHBoxLayout()
        self.he_name = QLineEdit("ur10_eob")
        self.he_btn = QPushButton("Start Calibration")
        self.he_btn.clicked.connect(self.toggle_handeye)
        self.he_status = QLabel("INACTIVE")
        self.he_status.setStyleSheet("color: gray; font-weight: bold;")
        he_layout.addWidget(QLabel("Name:"))
        he_layout.addWidget(self.he_name)
        he_layout.addWidget(self.he_btn)
        he_layout.addWidget(self.he_status)
        tools_layout.addLayout(he_layout)
        
        tools_group.setLayout(tools_layout)
        left_layout.addWidget(tools_group)
        
        # 3. Pose Manager
        pose_group = QGroupBox("3. Pose Sequence")
        pose_layout = QVBoxLayout()
        
        # Info
        self.pose_info = QLabel("Poses Loaded: 0")
        pose_layout.addWidget(self.pose_info)
        
        # File Selection
        self.btn_load_yaml = QPushButton("Load YAML Sequence")
        self.btn_load_yaml.clicked.connect(self.select_yaml_file)
        pose_layout.addWidget(self.btn_load_yaml)
        
        # Controls
        ctrl_layout = QHBoxLayout()
        self.btn_prev = QPushButton("<< Prev")
        self.btn_prev.clicked.connect(self.prev_pose)
        self.btn_next = QPushButton("Next >>")
        self.btn_next.clicked.connect(self.next_pose)
        self.btn_plan = QPushButton("Plan")
        self.btn_plan.clicked.connect(self.plan_pose)
        self.btn_plan.setStyleSheet("background-color: #FFC107; color: black; font-weight: bold;")
        
        self.btn_exec = QPushButton("Execute")
        self.btn_exec.clicked.connect(self.execute_pose)
        self.btn_exec.setStyleSheet("background-color: #2196F3; color: white; font-weight: bold;")

        ctrl_layout.addWidget(self.btn_prev)
        ctrl_layout.addWidget(self.btn_plan)
        ctrl_layout.addWidget(self.btn_exec)
        ctrl_layout.addWidget(self.btn_next)
        pose_layout.addLayout(ctrl_layout)
        
        # Record
        self.btn_record = QPushButton("Record Current Pose")
        self.btn_record.clicked.connect(self.record_pose)
        self.btn_record.setStyleSheet("background-color: #FF9800; color: white;")
        pose_layout.addWidget(self.btn_record)
        
        pose_group.setLayout(pose_layout)
        pose_group.setLayout(pose_layout)
        left_layout.addWidget(pose_group)
        
        # 4. Settings
        settings_group = QGroupBox("4. Settings")
        settings_layout = QVBoxLayout()
        
        # Controller Selector
        self.controller_combo = QComboBox()
        self.controller_combo.addItems([
            "/arm_controller/follow_joint_trajectory",
            "/joint_trajectory_controller/follow_joint_trajectory"
        ])
        self.controller_combo.setCurrentText("/arm_controller/follow_joint_trajectory")
        self.controller_combo.currentTextChanged.connect(self.change_controller)
        settings_layout.addWidget(QLabel("Trajectory Controller:"))
        settings_layout.addWidget(self.controller_combo)
        
        settings_group.setLayout(settings_layout)
        left_layout.addWidget(settings_group)
        
        left_layout.addStretch()
        main_layout.addLayout(left_layout, 1)
        
        # Right Panel: Logs
        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        main_layout.addWidget(self.log_output, 1)
        
        central_widget.setLayout(main_layout)

    def log_message(self, msg):
        self.log_output.append(msg)
        # Auto scroll
        sb = self.log_output.verticalScrollBar()
        sb.setValue(sb.maximum())

    def on_process_finished(self, name):
         if name in self.running_states: # It might be a oneshot process
            self.running_states[name] = False
         self.update_buttons()

    def update_buttons(self):
        # Robot
        if self.running_states['robot']:
            self.robot_btn.setText("Stop Robot")
            self.robot_btn.setStyleSheet("background-color: #f44336; color: white;")
            self.robot_status.setText("RUNNING")
            self.robot_status.setStyleSheet("color: green; font-weight: bold;")
        else:
            self.robot_btn.setText("Start Robot")
            self.robot_btn.setStyleSheet("background-color: #4CAF50; color: white;")
            self.robot_status.setText("INACTIVE")
            self.robot_status.setStyleSheet("color: gray; font-weight: bold;")
            
        # Camera
        if self.running_states['camera']:
            self.cam_btn.setText("Stop Camera")
            self.cam_btn.setStyleSheet("background-color: #f44336; color: white;")
            self.cam_status.setText("RUNNING")
            self.cam_status.setStyleSheet("color: green; font-weight: bold;")
        else:
            self.cam_btn.setText("Start Camera")
            self.cam_btn.setStyleSheet("background-color: #2196F3; color: white;")
            self.cam_status.setText("INACTIVE")
            self.cam_status.setStyleSheet("color: gray; font-weight: bold;")

        # MoveIt
        if self.running_states['moveit']:
            self.moveit_btn.setText("Stop MoveIt")
            self.moveit_status.setText("RUNNING")
            self.moveit_status.setStyleSheet("color: green; font-weight: bold;")
        else:
            self.moveit_btn.setText("Start MoveIt/RViz")
            self.moveit_status.setText("INACTIVE")
            self.moveit_status.setStyleSheet("color: gray; font-weight: bold;")

        # ArUco
        if self.running_states['aruco']:
            self.aruco_btn.setText("Stop ArUco")
            self.aruco_status.setText("RUNNING")
            self.aruco_status.setStyleSheet("color: green; font-weight: bold;")
        else:
            self.aruco_btn.setText("Start ArUco Node")
            self.aruco_status.setText("INACTIVE")
            self.aruco_status.setStyleSheet("color: gray; font-weight: bold;")

        # HandEye
        if self.running_states['handeye']:
            self.he_btn.setText("Stop Calibration")
            self.he_status.setText("RUNNING")
            self.he_status.setStyleSheet("color: green; font-weight: bold;")
        else:
             self.he_btn.setText("Start Calibration")
             self.he_status.setText("INACTIVE")
             self.he_status.setStyleSheet("color: gray; font-weight: bold;")

    # --- Actions ---
    def toggle_robot(self):
        if self.running_states['robot']:
            self.launch_manager.stop('robot')
        else:
            fake = self.robot_fake_chk.isChecked()
            ip = self.robot_ip.text()
            if not ip: ip = "192.168.11.100"

            # Piper arm launch via CAN bus
            cmd = "ros2 launch agilex_piper_description start_single_agx_arm.launch.py can_port:=can0 arm_type:=piper"
            
            self.launch_manager.start_process_thread('robot', cmd)
            self.running_states['robot'] = True
            
            self.controller_combo.setCurrentText("/arm_controller/follow_joint_trajectory")
            
        self.update_buttons()

    def activate_controller(self):
        if self.running_states['robot']:
            self.log_message("Attempting to auto-activate controller...")
            cmd = "ros2 control set_controller_state arm_controller active"
            subprocess.Popen(cmd, shell=True)


    def toggle_camera(self):
        if self.running_states['camera']:
            self.launch_manager.stop('camera')
        else:
             fake = self.cam_fake_chk.isChecked()
             if fake:
                 cmd = "ros2 run realsense2_camera realsense2_camera_node"  # No fake camera — use real RealSense
             else:
                 # 1280x720 = 720p
                 cmd = ("ros2 launch realsense2_camera rs_launch.py "
                        "rgb_camera.profile:=1280x720x30 "
                        "depth_module.profile:=1280x720x30 "
                        "align_depth.enable:=true")
             
             self.launch_manager.start_process_thread('camera', cmd)
             self.running_states['camera'] = True
        self.update_buttons()

    def toggle_moveit(self):
        if self.running_states['moveit']:
             self.launch_manager.stop('moveit')
        else:
             fake = self.moveit_fake_chk.isChecked()
             
             # Piper MoveIt launch
             cmd = "ros2 launch agilex_piper_description start_single_agx_arm_moveit.launch.py"
             
             self.launch_manager.start_process_thread('moveit', cmd)
             self.running_states['moveit'] = True
             
        self.update_buttons()
    
    def toggle_aruco(self):
        if self.running_states['aruco']:
             self.launch_manager.stop('aruco')
        else:
             # Aruco Launch (User Specified)
             cmd = ("ros2 run aruco_ros single --ros-args "
                    "-p marker_id:=582 "
                    "-p marker_size:=0.15 "
                    "-p reference_frame:=camera_color_optical_frame "
                    "-p camera_frame:=camera_color_optical_frame "
                    "-p marker_frame:=marker_frame "
                    "-r /camera_info:=/camera/camera/color/camera_info "
                    "-r /image:=/camera/camera/color/image_raw")
             self.launch_manager.start_process_thread('aruco', cmd)
             self.running_states['aruco'] = True
        self.update_buttons()

    def toggle_handeye(self):
        if self.running_states['handeye']:
             self.launch_manager.stop('handeye')
        else:
             name = self.he_name.text()
             if not name: name = "piper_eih"
             # eye_in_hand: camera is ON the arm (link6), marker is fixed in world
             cmd = (f"ros2 launch easy_handeye2 calibrate.launch.py name:={name} "
                    "calibration_type:=eye_in_hand "
                    "robot_base_frame:=base_link "
                    "robot_effector_frame:=link6 "
                    "tracking_base_frame:=camera_color_optical_frame "
                    "tracking_marker_frame:=marker_frame "
                    "freehand_robot_movement:=true")
             
             self.launch_manager.start_process_thread('handeye', cmd)
             self.running_states['handeye'] = True
        self.update_buttons()

    # --- Pose Logic ---
    def record_pose(self):
        if self.ros_node.current_joints is None:
            self.log_message("No joint states received! Is robot running?")
            return
            
        # Map current joints to standard UR10 order
        try:
            positions = []
            for name in self.ros_node.UR10_JOINTS:
                idx = self.ros_node.current_joints.name.index(name)
                positions.append(self.ros_node.current_joints.position[idx])
        except ValueError as e:
            self.log_message(f"Error mapping joints! Missing joint? {e}")
            return
            
        pose = {
            'pc': positions,
            'name': f"pose_{len(self.poses)+1}"
        }
        self.poses.append(pose)
        self.save_poses()
        self.log_message(f"Recorded Pose {len(self.poses)} (Order Corrected)")
        self.update_pose_ui()

    def save_poses(self):
        # Convert to list of dicts for better readability
        # Example:
        # poses:
        #   - name: pose_1
        #     joints: [0.1, 0.2, ...]
        data = []
        for p in self.poses:
            # Round for cleaner file
            rounded = [round(x, 4) for x in p['pc']]
            data.append({'name': p['name'], 'joints': rounded})
        
        with open(self.yaml_path, 'w') as f:
            # Custom dumper logic is complex, but PyYAML dump handles lists slightly messy.
            # Let's write manually or rely on flow style?
            # Or just use dump but accept it might expand.
            # Actually, dumping a dict with list usually expands unless we hint flow style.
            # Let's try default dump but structure it nicely.
            yaml.dump({'poses': data}, f, default_flow_style=None, sort_keys=False)

    def load_poses(self):
        if not os.path.exists(self.yaml_path):
             self.log_message(f"No pose file found at {self.yaml_path}")
             return
        
        with open(self.yaml_path, 'r') as f:
            data = yaml.safe_load(f)
            if data and 'poses' in data:
                self.poses = []
                poses_list = data.get('poses', [])
                if poses_list is None: poses_list = []
                
                for idx, item in enumerate(poses_list):
                    # Handle both old (list of lists) and new (list of dicts) formats
                    if isinstance(item, list):
                         # Old Format
                         self.poses.append({'pc': item, 'name': f"pose_{idx+1}"})
                    elif isinstance(item, dict) and 'joints' in item:
                         # New Format
                         name = item.get('name', f"pose_{idx+1}")
                         self.poses.append({'pc': item['joints'], 'name': name})
                
                self.log_message(f"Loaded {len(self.poses)} poses from YAML")
                self.update_pose_ui()

    def update_pose_ui(self):
        self.pose_info.setText(f"Poses: {len(self.poses)} | Current: {self.current_pose_idx+1}")
        
    def select_yaml_file(self):
        self.ros_node.get_logger().info("Button Clicked: Select YAML")
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Pose File", "", "YAML Files (*.yaml)")
        if file_path:
             self.yaml_path = file_path
             self.load_poses()
             self.log_message(f"Switched to sequence: {os.path.basename(file_path)}")
        else:
             self.ros_node.get_logger().info("No file selected in dialog")

    def change_controller(self, text):
        self.ros_node.update_controller(text)
        self.log_message(f"Switched controller to: {text}")

    def next_pose(self):
        self.ros_node.get_logger().info("Button Clicked: Move to Next Pose")
        if len(self.poses) == 0: 
            self.log_message("No poses loaded!")
            return
        self.current_pose_idx = (self.current_pose_idx + 1) % len(self.poses)
        self.update_pose_ui()
        # self.move_to_pose() # Disabled auto-move

    def prev_pose(self):
        if len(self.poses) == 0: return
        self.current_pose_idx = (self.current_pose_idx - 1) % len(self.poses)
        self.update_pose_ui()
        # self.move_to_pose() # Disabled auto-move
        
    def plan_pose(self):
        self.move_to_pose(plan_only=True)

    def execute_pose(self):
        self.move_to_pose(plan_only=False)

    def move_to_pose(self, plan_only=False):
        if not self.poses: return
        
        target = self.poses[self.current_pose_idx]
        action = "Planning" if plan_only else "Executing"
        self.log_message(f"{action} for {target['name']}...")
        
        future = self.ros_node.send_moveit_goal(target['pc'], plan_only=plan_only)
        
        if not future:
             self.log_message("Failed to send goal (MoveGroup Action Server unavailable?)")
             return

        future.add_done_callback(self.goal_response_callback)

    def feedback_callback(self, feedback_msg):
        # Optional: monitor feedback if needed
        # feedback = feedback_msg.feedback
        pass

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.log_message("Goal rejected :(")
            return
        self.log_message("Goal accepted! Executing...")
        res_future = goal_handle.get_result_async()
        res_future.add_done_callback(self.get_result_callback)

    def get_result_callback(self, future):
        result = future.result().result
        error_code = result.error_code.val
        if error_code == MoveItErrorCodes.SUCCESS:
             self.log_message("Move finished successfully.")
        else:
             self.log_message(f"Move failed with error code: {error_code}")

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    try:
        sys.exit(app.exec_())
    except Exception:
        pass
    finally:
        rclpy.shutdown()

if __name__ == '__main__':
    main()