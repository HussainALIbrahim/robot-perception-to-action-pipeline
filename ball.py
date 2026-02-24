# fixed_franka_vla_021.py
# ------------------------------------------------------------
# Your working collector + targeted fixes:
# 1) SAFE PARKING before placing objects (prevents initial bumps)
# 2) Single-task episodes (vision routing): cube->pick/place, ball->throw
# 3) 14 scripted demos, then 6 learned (BC) rollouts
# 4) Regenerator for bad episodes (17,18,19) -> forced scripted like 16
# ------------------------------------------------------------

import genesis as gs
import numpy as np
import cv2
import os
import json
from datetime import datetime

# Optional: PyTorch for imitation learning (behavior cloning)
_HAS_TORCH = True
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.optim.lr_scheduler import CosineAnnealingLR
except Exception as _e:
    _HAS_TORCH = False
    print(f"[imitation] PyTorch not available: {_e}. You can still collect data; training/inference requires torch.")


class FixedFrankaCollector:
    def __init__(self, output_dir="franka_dataset", use_ai_vision=True, yolo_weights="yolov8n.pt"):
        # Initialize Genesis ONCE
        gs.init(backend=gs.cpu, precision="32", debug=False)

        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(f"{self.output_dir}/episodes", exist_ok=True)
        os.makedirs(f"{self.output_dir}/images", exist_ok=True)
        os.makedirs(f"{self.output_dir}/bc_models", exist_ok=True)

        # Create scene
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=0.01),
            show_viewer=False
        )

        # Cache sim time step and action loop cadence
        self.dt = 0.01
        self.iter_steps_per_action = 6
        self.policy_iter_dt = self.dt * self.iter_steps_per_action

        # Add ground plane
        self.plane = self.scene.add_entity(gs.morphs.Plane())

        # Pre-create objects (we’ll reposition per episode)
        self.cube = self.scene.add_entity(
            gs.morphs.Box(size=(0.04, 0.04, 0.04), pos=(0.65, 0.0, 0.02))
        )
        self.ball = self.scene.add_entity(
            gs.morphs.Sphere(radius=0.028, pos=(0.65, -0.3, 0.028))
        )

        # Load Franka robot
        self.franka = self.scene.add_entity(
            gs.morphs.MJCF(file='xml/franka_emika_panda/panda.xml')
        )

        # Camera positioned to see full robot and workspace
        self.camera = self.scene.add_camera(
            res=(640, 480),
            pos=(2.0, 1.0, 1.2),
            lookat=(0.4, 0.0, 0.3),
            fov=60,
            GUI=False
        )

        # Build scene
        self.scene.build()

        # Robot control setup
        self.setup_robot_control()
        self.n_dofs = self.franka.n_dofs if hasattr(self.franka, "n_dofs") else len(self.franka.get_dofs_position())

        # Physics/material tweaks (guarded)
        try:
            if hasattr(self.cube, 'set_material_properties'):
                self.cube.set_material_properties(static_friction=1.6, dynamic_friction=1.2, restitution=0.0)
            if hasattr(self.plane, 'set_material_properties'):
                self.plane.set_material_properties(static_friction=1.4, dynamic_friction=1.15, restitution=0.0)
            if hasattr(self.ball, 'set_material_properties'):
                self.ball.set_material_properties(static_friction=1.35, dynamic_friction=1.15, restitution=0.03)
            if hasattr(self.ball, 'set_friction'):
                self.ball.set_friction(1.15)
            for fname in ['finger_link1', 'finger_link2', 'panda_finger_joint1', 'panda_finger_joint2', 'leftfinger', 'rightfinger', 'finger']:
                try:
                    link = self.franka.get_link(fname)
                    if link is not None and hasattr(link, 'set_material_properties'):
                        link.set_material_properties(static_friction=1.8, dynamic_friction=1.3, restitution=0.02)
                except Exception:
                    pass
        except Exception as e:
            print(f"[physics] Could not set custom material properties: {e}")

        # Vision controls
        self.use_ai_vision = bool(use_ai_vision)
        self.detector = None
        if self.use_ai_vision:
            try:
                from ultralytics import YOLO
                weights = yolo_weights if yolo_weights else "yolov8n.pt"
                self.detector = YOLO(weights)
                print(f"[vision] YOLO initialized with weights: {weights}")
            except Exception as e:
                print(f"[vision] YOLO not available; will use HoughCircles fallback only. Error: {e}")
                self.detector = None

        # Imitation learning (behavior cloning) setup
        self.bc_dir = os.path.join(self.output_dir, "bc_models")
        self.action_max_delta = np.array([0.055]*7 + [0.03]*2, dtype=np.float32)
        self.bc_models = {"cube": None, "ball": None}
        self.bc_stats = {"cube": None, "ball": None}

        # Gripper latch state
        self.grasp_latch = 0

    # ----------------- CORE SAFETY ADDITIONS -----------------

    def move_to_safe_parking(self):
        """
        NEW: Move the arm to a high/away parking pose BEFORE placing objects.
        This prevents initial accidental bumps (root cause for bad episodes 17–19).
        """
        quat = np.array([0, 1, 0, 0])
        safe_pos = np.array([0.45, 0.35, 0.60])  # high & +y (away from cube default)
        q_goal = self.franka.inverse_kinematics(link=self.end_effector, pos=safe_pos, quat=quat)
        if q_goal.shape[0] >= 9:
            q_goal[-2:] = 0.06  # fully open gripper
        path = self.franka.plan_path(qpos_goal=q_goal, num_waypoints=90)
        for wp in path:
            self.franka.control_dofs_position(wp, self.dofs_idx)
            self.scene.step()
        # extra settle
        for _ in range(60):
            self.scene.step()

    def setup_robot_control(self):
        """Set up proper robot control following Genesis documentation"""
        self.jnt_names = [
            'joint1', 'joint2', 'joint3', 'joint4',
            'joint5', 'joint6', 'joint7',
            'finger_joint1', 'finger_joint2'
        ]
        self.dofs_idx = [self.franka.get_joint(name).dof_idx_local for name in self.jnt_names]
        self.motors_dof = np.arange(7)
        self.fingers_dof = np.arange(7, 9)
        self.end_effector = self.franka.get_link('hand')

        # Softer arm Kp/Kv; higher finger gains
        self.franka.set_dofs_kp(
            kp=np.array([3200, 3200, 2600, 2600, 1600, 1600, 1600, 260, 260]),
            dofs_idx_local=self.dofs_idx
        )
        self.franka.set_dofs_kv(
            kv=np.array([420, 420, 340, 340, 220, 220, 220, 35, 35]),
            dofs_idx_local=self.dofs_idx
        )
        self.franka.set_dofs_force_range(
            lower=np.array([-87, -87, -87, -87, -12, -12, -12, -160, -160]),
            upper=np.array([ 87,  87,  87,  87,  12,  12,  12,  160,  160]),
            dofs_idx_local=self.dofs_idx
        )

    def reset_robot_pose(self):
        home_pose = np.array([0, -0.785, 0, -2.356, 0, 1.571, 0.785, 0.06, 0.06])
        self.franka.set_dofs_position(home_pose, self.dofs_idx)
        for _ in range(80):  # slightly longer settle
            self.scene.step()

    # ----------------- Scripted skills (expert) -----------------

    def pick_and_place_cube(self):
        quat = np.array([0, 1, 0, 0])

        # Via waypoint away from cube (positive y), then pre-grasp
        via_pos = np.array([0.65, 0.20, 0.30])
        q_via = self.franka.inverse_kinematics(link=self.end_effector, pos=via_pos, quat=quat)
        if q_via.shape[0] >= 9:
            q_via[-2:] = 0.06
        path = self.franka.plan_path(qpos_goal=q_via, num_waypoints=90)
        for wp in path:
            self.franka.control_dofs_position(wp, self.dofs_idx)
            self.scene.step()
            yield self.capture_frame("cube", "approach")

        pre_pos = np.array([0.65, 0.0, 0.30])
        q_pre = self.franka.inverse_kinematics(link=self.end_effector, pos=pre_pos, quat=quat)
        if q_pre.shape[0] >= 9:
            q_pre[-2:] = 0.06
        path = self.franka.plan_path(qpos_goal=q_pre, num_waypoints=80)
        for wp in path:
            self.franka.control_dofs_position(wp, self.dofs_idx)
            self.scene.step()
            yield self.capture_frame("cube", "approach")

        for _ in range(60):
            self.scene.step()
            yield self.capture_frame("cube", "stabilize")

        # Precise descend
        heights = np.linspace(0.30, 0.125, 60)
        for h in heights:
            qpos = self.franka.inverse_kinematics(link=self.end_effector, pos=np.array([0.65, 0.0, float(h)]), quat=quat)
            try:
                self.franka.control_dofs_position(qpos[:-2], self.motors_dof)
                self.franka.control_dofs_position(np.array([0.06, 0.06]), self.fingers_dof)
            except Exception:
                pass
            self.scene.step()
            yield self.capture_frame("cube", "lower")

        # Grasp dwell
        try:
            self.franka.control_dofs_position(np.array([0.0, 0.0]), self.fingers_dof)
        except Exception:
            pass
        try:
            self.franka.control_dofs_force(np.array([-0.9, -0.9]), self.fingers_dof)
        except Exception:
            pass
        for _ in range(150):
            self.scene.step()
            yield self.capture_frame("cube", "grasp")

        # Lift
        ee_now = self.end_effector.get_pos()
        if hasattr(ee_now, 'cpu'): ee_now = ee_now.cpu().numpy()
        lift_heights = np.linspace(ee_now[2], 0.28, 60)
        for h in lift_heights:
            qpos = self.franka.inverse_kinematics(link=self.end_effector, pos=np.array([0.65, 0.0, float(h)]), quat=quat)
            try:
                self.franka.control_dofs_position(qpos[:-2], self.motors_dof)
                self.franka.control_dofs_force(np.array([-0.9, -0.9]), self.fingers_dof)
            except Exception:
                pass
            self.scene.step()
            yield self.capture_frame("cube", "lift")

        # Transport
        qpos_goal = self.franka.inverse_kinematics(link=self.end_effector, pos=np.array([0.65, 0.4, 0.28]), quat=quat)
        path = self.franka.plan_path(qpos_goal=qpos_goal, num_waypoints=100)
        for waypoint in path:
            try:
                self.franka.control_dofs_position(waypoint[:-2], self.motors_dof)
                self.franka.control_dofs_force(np.array([-0.9, -0.9]), self.fingers_dof)
            except Exception:
                pass
            self.scene.step()
            yield self.capture_frame("cube", "transport")

        # Place & release
        qpos_place = self.franka.inverse_kinematics(link=self.end_effector, pos=np.array([0.65, 0.4, 0.13]), quat=quat)
        for _ in range(100):
            try:
                self.franka.control_dofs_position(qpos_place[:-2], self.motors_dof)
                self.franka.control_dofs_force(np.array([-0.9, -0.9]), self.fingers_dof)
            except Exception:
                pass
            self.scene.step()
            yield self.capture_frame("cube", "place")

        try:
            self.franka.control_dofs_force(np.array([0.0, 0.0]), self.fingers_dof)
        except Exception:
            pass
        try:
            self.franka.control_dofs_position(np.array([0.06, 0.06]), self.fingers_dof)
        except Exception:
            pass
        for _ in range(100):
            self.scene.step()
            yield self.capture_frame("cube", "release")

    def throw_ball(self):
        quat = np.array([0, 1, 0, 0])

        # Approach
        ball_pos = self.ball.get_pos()
        if hasattr(ball_pos, 'cpu'): ball_pos = ball_pos.cpu().numpy()
        qpos = self.franka.inverse_kinematics(link=self.end_effector, pos=np.array([ball_pos[0], ball_pos[1], 0.25]), quat=quat)
        if qpos.shape[0] >= 9: qpos[-2:] = 0.06
        path = self.franka.plan_path(qpos_goal=qpos, num_waypoints=100)
        for waypoint in path:
            self.franka.control_dofs_position(waypoint, self.dofs_idx)
            self.scene.step()
            yield self.capture_frame("ball", "approach")
        for _ in range(120):
            self.scene.step()
            yield self.capture_frame("ball", "stabilize")

        # Descend & grasp
        lock_ball = self.ball.get_pos()
        if hasattr(lock_ball, 'cpu'): lock_ball = lock_ball.cpu().numpy()
        approach_heights = np.linspace(0.25, 0.13, 60)
        for h in approach_heights:
            qpos = self.franka.inverse_kinematics(link=self.end_effector, pos=np.array([lock_ball[0], lock_ball[1], float(h)]), quat=quat)
            try:
                self.franka.control_dofs_position(qpos[:-2], self.motors_dof)
                self.franka.control_dofs_position(np.array([0.06, 0.06]), self.fingers_dof)
            except Exception:
                pass
            self.scene.step()
            yield self.capture_frame("ball", "lower")

        try:
            self.franka.control_dofs_position(np.array([0.0, 0.0]), self.fingers_dof)
        except Exception:
            pass
        try:
            self.franka.control_dofs_force(np.array([-1.2, -1.2]), self.fingers_dof)
        except Exception:
            pass
        for _ in range(150):
            self.scene.step()
            yield self.capture_frame("ball", "grasp_force")

        # Lift to throw ready
        ee_now = self.end_effector.get_pos()
        if hasattr(ee_now, 'cpu'): ee_now = ee_now.cpu().numpy()
        lift_heights = np.linspace(ee_now[2], 0.40, 50)
        for h in lift_heights:
            qpos = self.franka.inverse_kinematics(link=self.end_effector, pos=np.array([ee_now[0], ee_now[1], float(h)]), quat=quat)
            try:
                self.franka.control_dofs_position(qpos[:-2], self.motors_dof)
                self.franka.control_dofs_force(np.array([-1.2, -1.2]), self.fingers_dof)
            except Exception:
                pass
            self.scene.step()
            yield self.capture_frame("ball", "lift")

        # Wind-up
        wind_up_pos = np.array([0.5, -0.2, 0.5])
        qpos = self.franka.inverse_kinematics(link=self.end_effector, pos=wind_up_pos, quat=quat)
        path = self.franka.plan_path(qpos_goal=qpos, num_waypoints=100)
        for waypoint in path:
            try:
                self.franka.control_dofs_position(waypoint[:-2], self.motors_dof)
                self.franka.control_dofs_force(np.array([-1.2, -1.2]), self.fingers_dof)
            except Exception:
                pass
            self.scene.step()
            yield self.capture_frame("ball", "wind_up")

        # Clear forces before throw (clean release)
        try:
            self.franka.control_dofs_force(np.array([0.0, 0.0]), self.fingers_dof)
        except Exception:
            pass

        # Throw sweep + release window
        throw_pos = np.array([0.8, 0.3, 0.3])
        qpos = self.franka.inverse_kinematics(link=self.end_effector, pos=throw_pos, quat=quat)
        path = self.franka.plan_path(qpos_goal=qpos, num_waypoints=55)
        start_release = int(len(path) * 0.60)
        end_release   = int(len(path) * 0.75)
        for i, waypoint in enumerate(path):
            if waypoint.shape[0] >= 9:
                waypoint[-2:] = 0.0 if i < start_release else 0.06
            try:
                self.franka.control_dofs_force(np.array([0.0, 0.0]), self.fingers_dof)
            except Exception:
                pass
            self.franka.control_dofs_position(waypoint, self.dofs_idx)
            self.scene.step()
            yield self.capture_frame("ball", "throwing")

        # Follow-through
        for i in range(60):
            follow_pos = np.array([0.85, 0.35, 0.25 if i < 30 else 0.2])
            qpos = self.franka.inverse_kinematics(link=self.end_effector, pos=follow_pos, quat=quat)
            if qpos.shape[0] >= 9:
                qpos[-2:] = 0.06
            self.franka.control_dofs_position(qpos, self.dofs_idx)
            self.scene.step()
            yield self.capture_frame("ball", "follow_through")

    def throw_from_current_hold(self):
        # (kept from your original)
        quat = np.array([0, 1, 0, 0])
        ee = self._get_ee_pos_np()
        qpos = self.franka.inverse_kinematics(link=self.end_effector,
                                              pos=np.array([ee[0], ee[1], max(ee[2], 0.40)]),
                                              quat=quat)
        path = self.franka.plan_path(qpos_goal=qpos, num_waypoints=35)
        for wp in path:
            self.franka.control_dofs_position(wp[:-2], self.motors_dof)
            try:
                self.franka.control_dofs_force(np.array([-1.2, -1.2]), self.fingers_dof)
            except Exception:
                pass
            self.scene.step()
            yield self.capture_frame("ball", "lift")

        wind_up_pos = np.array([0.5, -0.2, 0.5])
        qpos = self.franka.inverse_kinematics(link=self.end_effector, pos=wind_up_pos, quat=quat)
        path = self.franka.plan_path(qpos_goal=qpos, num_waypoints=50)
        for wp in path:
            self.franka.control_dofs_position(wp[:-2], self.motors_dof)
            try:
                self.franka.control_dofs_force(np.array([-1.2, -1.2]), self.fingers_dof)
            except Exception:
                pass
            self.scene.step()
            yield self.capture_frame("ball", "wind_up")

        try:
            self.franka.control_dofs_force(np.array([0.0, 0.0]), self.fingers_dof)
        except Exception:
            pass

        throw_pos = np.array([0.8, 0.3, 0.3])
        qpos = self.franka.inverse_kinematics(link=self.end_effector, pos=throw_pos, quat=quat)
        path = self.franka.plan_path(qpos_goal=qpos, num_waypoints=55)
        start_release = int(len(path) * 0.60)
        for i, wp in enumerate(path):
            if wp.shape[0] >= 9:
                wp[-2:] = 0.0 if i < start_release else 0.06
            try:
                self.franka.control_dofs_force(np.array([0.0, 0.0]), self.fingers_dof)
            except Exception:
                pass
            self.franka.control_dofs_position(wp, self.dofs_idx)
            self.scene.step()
            yield self.capture_frame("ball", "throwing")

        for i in range(40):
            follow_pos = np.array([0.85, 0.35, 0.25 if i < 20 else 0.2])
            qpos = self.franka.inverse_kinematics(link=self.end_effector, pos=follow_pos, quat=quat)
            if qpos.shape[0] >= 9:
                qpos[-2:] = 0.06
            self.franka.control_dofs_position(qpos, self.dofs_idx)
            self.scene.step()
            yield self.capture_frame("ball", "follow_through")

    # ---------- Vision: detection and routing ----------

    def capture_frame(self, task, phase):
        try:
            rr = self.camera.render(rgb=True, depth=False, segmentation=False)
            if rr is None:
                return None
            if hasattr(rr, 'rgb'):
                rgb_data = rr.rgb
            elif isinstance(rr, (tuple, list)) and len(rr) > 0:
                rgb_data = rr[0]
            else:
                rgb_data = rr
            if hasattr(rgb_data, 'cpu'):
                rgb_data = rgb_data.cpu().numpy()
            rgb_data = np.asarray(rgb_data)
            if rgb_data.dtype != np.uint8:
                if np.issubdtype(rgb_data.dtype, np.floating):
                    rgb_data = (np.clip(rgb_data, 0.0, 1.0) * 255.0).astype(np.uint8) if rgb_data.max() <= 1.0 else np.clip(rgb_data, 0, 255).astype(np.uint8)
                else:
                    rgb_data = np.clip(rgb_data, 0, 255).astype(np.uint8)

            current_joints = self.franka.get_dofs_position()
            if hasattr(current_joints, 'cpu'):
                current_joints = current_joints.cpu().numpy()

            cube_pos = self.cube.get_pos()
            ball_pos = self.ball.get_pos()
            if hasattr(cube_pos, 'cpu'):
                cube_pos = cube_pos.cpu().numpy()
            if hasattr(ball_pos, 'cpu'):
                ball_pos = ball_pos.cpu().numpy()

            ee_pos, ee_quat = None, None
            try:
                ee_p = self.end_effector.get_pos()
                ee_q = self.end_effector.get_quat() if hasattr(self.end_effector, "get_quat") else None
                if hasattr(ee_p, 'cpu'): ee_p = ee_p.cpu().numpy()
                if ee_q is not None and hasattr(ee_q, 'cpu'): ee_q = ee_q.cpu().numpy()
                ee_pos = np.asarray(ee_p).tolist()
                ee_quat = np.asarray(ee_q).tolist() if ee_q is not None else None
            except Exception:
                pass

            return {
                'image': rgb_data,
                'task': task,
                'phase': phase,
                'robot_joints': np.asarray(current_joints).tolist(),
                'cube_pos': np.asarray(cube_pos).tolist(),
                'ball_pos': np.asarray(ball_pos).tolist(),
                'ee_pos': ee_pos,
                'ee_quat': ee_quat,
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            print(f"Error capturing frame: {e}")
        return None

    def detect_object_shape(self, rgb_image):
        # YOLO if available, else HoughCircles
        try:
            if rgb_image is None:
                return "cube"
            img = rgb_image
            if hasattr(img, 'cpu'):
                img = img.cpu().numpy()
            img = np.asarray(img)
            if img.ndim == 3 and img.shape[2] == 4:
                img = img[..., :3]
            if img.dtype != np.uint8:
                if np.issubdtype(img.dtype, np.floating):
                    img = (np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8) if img.max() <= 1.0 else np.clip(img, 0, 255).astype(np.uint8)
                else:
                    img = np.clip(img, 0, 255).astype(np.uint8)

            if self.detector is not None:
                try:
                    results = self.detector.predict(source=img, stream=False, verbose=False)
                    if results and len(results) > 0:
                        res = results[0]
                        names = getattr(self.detector, 'names', None) or getattr(res, 'names', None)
                        def name_for(c):
                            try:
                                if isinstance(names, dict):
                                    return names.get(int(c), str(c))
                                elif isinstance(names, (list, tuple)):
                                    return names[int(c)]
                            except Exception:
                                return str(int(c))
                            return str(int(c))
                        boxes = getattr(res, 'boxes', None)
                        if boxes is not None and getattr(boxes, 'cls', None) is not None:
                            cls = boxes.cls.cpu().numpy()
                            conf = boxes.conf.cpu().numpy() if getattr(boxes, 'conf', None) is not None else [0.0] * len(cls)
                            labels = [str(name_for(c)).lower() for c in cls]
                            ball_syns = {"sports ball", "ball", "tennis ball", "basketball", "soccer ball", "baseball", "volleyball", "football"}
                            for lab, cf in zip(labels, conf):
                                if any(s in lab for s in ball_syns) and cf >= 0.38:
                                    return "ball"
                except Exception as ye:
                    print(f"[vision] YOLO predict error: {ye}")

            bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (9, 9), 2)
            circles = cv2.HoughCircles(
                gray, cv2.HOUGH_GRADIENT, dp=1.3, minDist=30,
                param1=100, param2=20, minRadius=5, maxRadius=150
            )
            if circles is not None and len(circles) > 0:
                return "ball"
        except Exception as e:
            print(f"[vision] detection error: {e}")
        return "cube"

    def detect_and_route(self):
        try:
            rr = self.camera.render(rgb=True, depth=False, segmentation=False)
            if rr is None:
                return "cube"
            if hasattr(rr, 'rgb'):
                rgb = rr.rgb
            elif isinstance(rr, (tuple, list)) and len(rr) > 0:
                rgb = rr[0]
            else:
                rgb = rr
            if hasattr(rgb, 'cpu'):
                rgb = rgb.cpu().numpy()
            rgb = np.asarray(rgb)
            if rgb.dtype != np.uint8:
                if np.issubdtype(rgb.dtype, np.floating):
                    rgb = (np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8) if rgb.max() <= 1.0 else np.clip(rgb, 0, 255).astype(np.uint8)
            return self.detect_object_shape(rgb)
        except Exception as e:
            print(f"[vision] detect_and_route error: {e}")
            return "cube"

    # ---------- Helper: goal-relative state + warm start ----------

    def _get_ee_pos_np(self):
        ee = self.end_effector.get_pos()
        if hasattr(ee, 'cpu'):
            ee = ee.cpu().numpy()
        return np.asarray(ee, dtype=np.float32)

    def _target_pos(self, task):
        tgt = self.cube.get_pos() if task == "cube" else self.ball.get_pos()
        if hasattr(tgt, 'cpu'):
            tgt = tgt.cpu().numpy()
        return np.asarray(tgt, dtype=np.float32)

    def _warm_start_for_task(self, task):
        try:
            if task == "cube":
                target_pos = np.array([0.65, 0.0, 0.25])
            else:
                ball_pos = self._target_pos("ball")
                target_pos = np.array([ball_pos[0], ball_pos[1], 0.25])
            qpos = self.franka.inverse_kinematics(link=self.end_effector, pos=target_pos, quat=np.array([0, 1, 0, 0]))
            if qpos.shape[0] >= 9:
                qpos[-2:] = 0.06
            path = self.franka.plan_path(qpos_goal=qpos, num_waypoints=120)
            for waypoint in path:
                self.franka.control_dofs_position(waypoint, self.dofs_idx)
                self.scene.step()
                yield self.capture_frame(task, "warm_start_approach")
            for _ in range(80):
                self.scene.step()
                yield self.capture_frame(task, "warm_start_settle")
        except Exception as e:
            print(f"[policy] Warm-start error: {e}")
            return

    # ---------- Episode recording (UPDATED FLOW) ----------

    def _next_episode_index(self):
        eps_dir = os.path.join(self.output_dir, "episodes")
        os.makedirs(eps_dir, exist_ok=True)
        ids = []
        for fn in os.listdir(eps_dir):
            if fn.startswith("episode_") and fn.endswith(".json"):
                try:
                    ids.append(int(fn.split("_")[-1].split(".")[0]))
                except Exception:
                    pass
        return 0 if not ids else max(ids) + 1

    def record_episode(self, episode_id, use_bc_policy=False):
        print(f"\n=== Recording episode {episode_id} ===")
        # 1) Reset robot & go to safe parking BEFORE placing objects
        self.reset_robot_pose()
        self.move_to_safe_parking()   # <<< BIG FIX

        # 2) Place objects (farther ball to reduce incidental bumps)
        self.cube.set_pos([0.65, 0.0, 0.02])
        self.ball.set_pos([0.65, -0.45, 0.025])

        # 3) Let physics settle a bit
        for _ in range(50):
            self.scene.step()

        # 4) Vision routing (single-task per episode)
        route = self.detect_and_route() if self.use_ai_vision else "cube"
        print(f"[router] task = {route}")

        episode_data = {
            "episode_id": episode_id,
            "physics_type": "ik_pd",
            "robot_control": "ik_pd + optional_behavior_cloning",
            "frames": [],
            "task": route
        }
        successful_images = 0
        step_count = 0

        def save_frame(frame_data, task_label):
            nonlocal successful_images, step_count
            if frame_data:
                img_filename = f"ep{episode_id:03d}_step{step_count:03d}.png"
                img_path = os.path.join(self.output_dir, "images", img_filename)
                bgr = cv2.cvtColor(frame_data['image'], cv2.COLOR_RGB2BGR)
                ok = cv2.imwrite(img_path, bgr)
                if ok:
                    frame_data["image_path"] = img_filename
                    frame_data["step"] = step_count
                    del frame_data['image']
                    episode_data["frames"].append(frame_data)
                    successful_images += 1
                step_count += 1
                if step_count % 50 == 0:
                    print(f"    {task_label}: {step_count} frames")

        def run_bc_if_available(task):
            model_loaded = self.load_bc_policy(task)
            if use_bc_policy and model_loaded:
                print(f"[policy] Running BC policy for task: {task}")
                for f in self._warm_start_for_task(task):
                    save_frame(f, f"{task} warm start")
                for f in self.run_bc_policy(task, max_steps=600):
                    save_frame(f, f"{task} BC policy")
                return True
            return False

        # 5) Execute exactly ONE task per episode (based on vision)
        if route == "ball":
            if not run_bc_if_available("ball"):
                for f in self.throw_ball(): save_frame(f, "Ball throwing")
        else:
            if not run_bc_if_available("cube"):
                for f in self.pick_and_place_cube(): save_frame(f, "Cube manipulation")

        # 6) Save episode JSON
        ep_file = os.path.join(self.output_dir, "episodes", f"episode_{episode_id:03d}.json")
        with open(ep_file, 'w') as f:
            json.dump(episode_data, f, indent=2)
        print(f"Episode {episode_id} saved: {successful_images} images, {step_count} steps")
        return episode_data

    def collect_fixed_dataset(self, num_episodes=2, use_bc_policy=False):
        all_episodes = []
        start_idx = self._next_episode_index()
        for k in range(num_episodes):
            eid = start_idx + k
            ep = self.record_episode(eid, use_bc_policy=use_bc_policy)
            all_episodes.append(ep)
        return all_episodes

    # ---------- Imitation Learning (Behavior Cloning) ----------

    def _task_one_hot(self, task):
        return np.array([1.0, 0.0], dtype=np.float32) if task == "cube" else np.array([0.0, 1.0], dtype=np.float32)

    def _obs_from_live(self, task):
        joints = self.franka.get_dofs_position()
        if hasattr(joints, 'cpu'):
            joints = joints.cpu().numpy()
        joints = np.asarray(joints, dtype=np.float32)
        if joints.shape[0] >= 9:
            joints = joints[:9].astype(np.float32)
        cube_pos = self.cube.get_pos(); ball_pos = self.ball.get_pos()
        if hasattr(cube_pos, 'cpu'): cube_pos = cube_pos.cpu().numpy()
        if hasattr(ball_pos, 'cpu'): ball_pos = ball_pos.cpu().numpy()
        ee_pos = self._get_ee_pos_np()
        target = self._target_pos(task)
        ee_to_target = (target - ee_pos).astype(np.float32)
        obs = np.concatenate([
            joints.astype(np.float32),
            np.asarray(cube_pos, dtype=np.float32),
            np.asarray(ball_pos, dtype=np.float32),
            ee_pos.astype(np.float32),
            ee_to_target.astype(np.float32),
            self._task_one_hot(task)
        ], axis=0)
        return obs

    def _obs_from_frame(self, frame):
        joints = np.asarray(frame["robot_joints"], dtype=np.float32)
        if joints.shape[0] >= 9:
            joints = joints[:9]
        cube_pos = np.asarray(frame["cube_pos"], dtype=np.float32)
        ball_pos = np.asarray(frame["ball_pos"], dtype=np.float32)
        ee_pos = np.asarray(frame.get("ee_pos", [0,0,0]), dtype=np.float32)
        task = frame["task"]
        target = cube_pos if task == "cube" else ball_pos
        ee_to_target = (target - ee_pos).astype(np.float32)
        return np.concatenate([joints, cube_pos, ball_pos, ee_pos, ee_to_target, self._task_one_hot(task)], axis=0)

    def _find_episode_files(self):
        eps_dir = os.path.join(self.output_dir, "episodes")
        files = []
        if os.path.isdir(eps_dir):
            for fn in os.listdir(eps_dir):
                if fn.endswith(".json"):
                    files.append(os.path.join(eps_dir, fn))
        return sorted(files)

    def _build_bc_dataset(self, task, frame_stride=3, k_step=3):
        files = self._find_episode_files()
        X_list, Y_list = [], []
        n_pairs = 0
        keep = {"approach","lower","grasp","grasp_force","lift","transport","place","release",
                "warm_start_approach","warm_start_settle","wind_up","throwing","follow_through"}
        for ep_file in files:
            try:
                with open(ep_file, 'r') as f:
                    ep = json.load(f)
            except Exception as e:
                print(f"[imitation] Failed to read {ep_file}: {e}")
                continue
            frames = [fr for fr in ep.get("frames", []) if fr.get("task") == task and fr.get("phase") in keep]
            if len(frames) < (k_step + 1):
                continue
            idxs = list(range(0, len(frames)-k_step, frame_stride))
            for i in idxs:
                f0 = frames[i]
                f1 = frames[i + k_step]
                x = self._obs_from_frame(f0)
                q0 = np.asarray(f0["robot_joints"], dtype=np.float32)[:9]
                q1 = np.asarray(f1["robot_joints"], dtype=np.float32)[:9]
                y_delta = q1 - q0
                y_norm = np.clip(y_delta / self.action_max_delta, -1.0, 1.0).astype(np.float32)
                X_list.append(x.astype(np.float32))
                Y_list.append(y_norm)
                n_pairs += 1
        if n_pairs == 0:
            print(f"[imitation] No training pairs found for task '{task}'. Collect dataset first.")
            return None, None, None
        X = np.stack(X_list, axis=0).astype(np.float32)
        Y = np.stack(Y_list, axis=0).astype(np.float32)
        x_mean = X.mean(axis=0)
        x_std = X.std(axis=0) + 1e-6
        stats = {"x_mean": x_mean.tolist(), "x_std": x_std.tolist()}
        print(f"[imitation] Built dataset for '{task}': {X.shape[0]} samples, obs_dim={X.shape[1]}")
        return X, Y, stats

    def _save_bc_stats(self, task, stats):
        os.makedirs(self.bc_dir, exist_ok=True)
        path = os.path.join(self.bc_dir, f"bc_{task}_stats.json")
        with open(path, 'w') as f:
            json.dump(stats, f, indent=2)

    def _load_bc_stats(self, task):
        path = os.path.join(self.bc_dir, f"bc_{task}_stats.json")
        if not os.path.isfile(path):
            return None
        with open(path, 'r') as f:
            return json.load(f)

    def _bc_model_path(self, task):
        return os.path.join(self.bc_dir, f"bc_{task}.pt")

    def train_bc_model(self, task, epochs=150, batch_size=256, lr=1e-3, hidden_sizes=(256, 256), patience=10):
        if not _HAS_TORCH:
            print("[imitation] PyTorch not installed. Please `pip install torch` to train.")
            return False
        X, Y, stats = self._build_bc_dataset(task, frame_stride=3, k_step=3)
        if X is None:
            return False
        x_mean = np.asarray(stats["x_mean"], dtype=np.float32)
        x_std = np.asarray(stats["x_std"], dtype=np.float32)
        self._save_bc_stats(task, stats)
        N = X.shape[0]
        idx = np.random.permutation(N)
        n_val = max(1, min(max(50, int(0.1 * N)), N - 1))
        val_idx = idx[:n_val]; train_idx = idx[n_val:]
        X_train = ((X[train_idx] - x_mean) / x_std); Y_train = Y[train_idx]
        X_val = ((X[val_idx] - x_mean) / x_std); Y_val = Y[val_idx]
        device = torch.device("cpu")
        input_dim = X.shape[1]; output_dim = Y.shape[1]
        model = nn.Sequential(
            nn.Linear(input_dim, hidden_sizes[0]), nn.ReLU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]), nn.ReLU(),
            nn.Linear(hidden_sizes[1], output_dim), nn.Tanh()
        ).to(device)
        opt = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        loss_fn = nn.MSELoss()
        X_train_t = torch.from_numpy(X_train).to(device); Y_train_t = torch.from_numpy(Y_train).to(device)
        X_val_t = torch.from_numpy(X_val).to(device); Y_val_t = torch.from_numpy(Y_val).to(device)
        steps_per_epoch = max(1, int(np.ceil(X_train_t.shape[0] / batch_size)))
        scheduler = CosineAnnealingLR(opt, T_max=epochs)
        print(f"[imitation] Training BC for '{task}' | epochs={epochs} | train={X_train_t.shape[0]} | val={X_val_t.shape[0]}")
        best_val = float('inf'); best_ep = 0
        for ep in range(1, epochs + 1):
            model.train(); perm = torch.randperm(X_train_t.shape[0]); train_loss_sum = 0.0
            for b in range(steps_per_epoch):
                bi = perm[b*batch_size: (b+1)*batch_size]
                xb = X_train_t[bi]; yb = Y_train_t[bi]
                pred = model(xb); loss = loss_fn(pred, yb)
                opt.zero_grad(); loss.backward(); opt.step()
                train_loss_sum += loss.item() * xb.shape[0]
            train_loss = train_loss_sum / max(1, X_train_t.shape[0])
            model.eval()
            with torch.no_grad():
                val_pred = model(X_val_t); val_loss = loss_fn(val_pred, Y_val_t).item()
            scheduler.step()
            if val_loss < best_val:
                best_val = val_loss; best_ep = ep
                torch.save({"state_dict": model.state_dict(),
                            "input_dim": input_dim,
                            "output_dim": output_dim,
                            "hidden": list(hidden_sizes)}, self._bc_model_path(task))
            if ep % 5 == 0 or ep == 1 or ep == epochs:
                print(f"  [ep {ep:03d}] train_mse={train_loss:.6f} | val_mse={val_loss:.6f} | lr={scheduler.get_last_lr()[0]:.2e}")
            if ep - best_ep >= patience:
                print(f"  [early stop] no val improvement for {patience} epochs (best at ep {best_ep})")
                break
        print(f"[imitation] Saved best BC model for '{task}' to {self._bc_model_path(task)} (best_val={best_val:.6f})")
        return True

    def train_bc_models_from_dataset(self, tasks=("cube", "ball"), epochs=150):
        ok_any = False
        for t in tasks:
            ok = self.train_bc_model(t, epochs=epochs)
            ok_any = ok_any or ok
        if not ok_any:
            print("[imitation] No models trained (insufficient data?).")
        return ok_any

    def load_bc_policy(self, task):
        if not _HAS_TORCH:
            return False
        model_path = self._bc_model_path(task)
        stats = self._load_bc_stats(task)
        if not os.path.isfile(model_path) or stats is None:
            print(f"[policy] No BC model/stats for '{task}'.")
            return False
        ckpt = torch.load(model_path, map_location="cpu")
        input_dim = ckpt.get("input_dim", None)
        output_dim = ckpt.get("output_dim", 9)
        hidden = ckpt.get("hidden", [256, 256])
        model = nn.Sequential(
            nn.Linear(input_dim, hidden[0]), nn.ReLU(),
            nn.Linear(hidden[0], hidden[1]), nn.ReLU(),
            nn.Linear(hidden[1], output_dim), nn.Tanh()
        )
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        self.bc_models[task] = model
        self.bc_stats[task] = stats
        print(f"[policy] Loaded BC model for '{task}' from {model_path}")
        return True

    # ---------- Hybrid BC runtime (unchanged from your logic) ----------

    def run_bc_policy(self, task, max_steps=600):
        if not _HAS_TORCH:
            print("[policy] PyTorch not installed; cannot run policy.")
            return
        model = self.bc_models.get(task, None)
        stats = self.bc_stats.get(task, None)
        if model is None or stats is None:
            if not self.load_bc_policy(task):
                return
            model = self.bc_models.get(task, None)
            stats = self.bc_stats.get(task, None)
            if model is None or stats is None:
                print(f"[policy] Missing model/stats for '{task}' after load; aborting BC run.")
                return
        if not isinstance(stats, dict) or "x_mean" not in stats or "x_std" not in stats:
            print(f"[policy] Stats missing keys for '{task}'; aborting BC run.")
            return
        x_mean = np.asarray(stats["x_mean"], dtype=np.float32)
        x_std = np.asarray(stats["x_std"], dtype=np.float32)
        action_max = self.action_max_delta.astype(np.float32)

        phase = "approach"
        close_done_steps = 0

        if task == "ball":
            near_xy_thresh = 0.12; near_z_thresh = 0.05; latch_duration_s = 1.2; min_progress = 0.0006; stagnation_limit = 25
        else:
            near_xy_thresh = 0.06; near_z_thresh = 0.035; latch_duration_s = 0.5; min_progress = 0.0010; stagnation_limit = 35
        latch_steps_default = max(2, int(latch_duration_s / max(1e-4, self.policy_iter_dt)))

        ball_target_lock = None
        prev_dist = None; stagnation_steps = 0

        for step in range(max_steps):
            obs = self._obs_from_live(task).astype(np.float32)
            x_norm = (obs - x_mean) / x_std
            x_t = torch.from_numpy(x_norm[None, :]).to("cpu")
            with torch.no_grad():
                y_norm = model(x_t).cpu().numpy()[0]
            base_delta = np.clip(y_norm, -1.0, 1.0) * action_max

            qcur = self.franka.get_dofs_position()
            if hasattr(qcur, 'cpu'): qcur = qcur.cpu().numpy()
            qcur = np.asarray(qcur, dtype=np.float32)

            ee = self._get_ee_pos_np()
            tgt_raw = self._target_pos(task)
            if task == "ball":
                dxy_tmp = float(np.linalg.norm(np.array(ee[:2]) - np.array(tgt_raw[:2])))
                if ball_target_lock is None and dxy_tmp < 0.15:
                    ball_target_lock = tgt_raw.copy()
            tgt = ball_target_lock if (task == "ball" and ball_target_lock is not None) else tgt_raw

            dxy = float(np.linalg.norm(np.array(ee[:2]) - np.array(tgt[:2])))
            dz = float(abs(ee[2] - tgt[2]))
            dist3d = float(np.linalg.norm(ee - tgt))

            if task == "cube":
                if phase == "approach":
                    if dxy < 0.06: phase = "descend"
                elif phase == "descend":
                    if dz < 0.05 and dxy < 0.06: phase = "close"
                elif phase == "close":
                    try: self.franka.control_dofs_force(np.array([-0.9, -0.9]), self.fingers_dof)
                    except Exception: pass
                    close_done_steps += 1
                    if close_done_steps > 40: phase = "lift"
                elif phase == "lift":
                    try:
                        lift_q = self.franka.inverse_kinematics(link=self.end_effector,
                            pos=np.array([ee[0], ee[1], max(ee[2] + 0.01, tgt[2] + 0.12)]),
                            quat=np.array([0, 1, 0, 0]))
                        self.franka.control_dofs_position(lift_q[:7], self.motors_dof)
                    except Exception: pass
            else:
                if phase == "approach":
                    if dxy < 0.10: phase = "descend"
                elif phase == "descend":
                    if dz < 0.06 and dxy < 0.12: phase = "close"
                elif phase == "close":
                    try: self.franka.control_dofs_force(np.array([-2.0, -2.0]), self.fingers_dof)
                    except Exception: pass
                    close_done_steps += 1
                    if close_done_steps > 50: phase = "lift"
                elif phase == "lift":
                    try:
                        lift_q = self.franka.inverse_kinematics(link=self.end_effector,
                            pos=np.array([ee[0], ee[1], max(ee[2] + 0.01, tgt[2] + 0.10)]),
                            quat=np.array([0, 1, 0, 0]))
                        self.franka.control_dofs_position(lift_q[:7], self.motors_dof)
                    except Exception: pass
                    if ee[2] > (tgt[2] + 0.12):
                        print("[policy] Throw-ready height. Using expert throw from hold.")
                        for frame in self.throw_from_current_hold(): yield frame
                        return

            if dxy < near_xy_thresh and dz < near_z_thresh and self.grasp_latch == 0 and phase not in ("close", "lift"):
                self.grasp_latch = latch_steps_default

            if self.grasp_latch > 0:
                try: self.franka.control_dofs_force(np.array([-3.0, -3.0]), self.fingers_dof)
                except Exception: pass
                self.grasp_latch -= 1
            else:
                if task == "cube" and phase in ("close", "lift"):
                    try: self.franka.control_dofs_force(np.array([-0.9, -0.9]), self.fingers_dof)
                    except Exception: pass
                elif task == "ball":
                    if dxy < 0.10 and dz < 0.06:
                        try: self.franka.control_dofs_force(np.array([-2.0, -2.0]), self.fingers_dof)
                        except Exception: pass
                    else:
                        try: self.franka.control_dofs_position(np.array([0.06, 0.06]), self.fingers_dof)
                        except Exception: pass
                else:
                    try: self.franka.control_dofs_position(np.array([0.06, 0.06]), self.fingers_dof)
                    except Exception: pass

            delta_used = base_delta.copy()
            if task == "ball" and (dxy < 0.12 and dz < 0.08):
                scale = 0.6
                if dxy < 0.08 and dz < 0.05: scale = 0.4
                delta_used *= scale
            if task == "cube" and (dxy < 0.08 and dz < 0.06):
                delta_used *= 0.6

            qcmd = qcur.copy()
            if qcur.shape[0] >= 9:
                qcmd[:9] = qcmd[:9] + delta_used
            else:
                qcmd[:qcur.shape[0]] = qcmd[:qcur.shape[0]] + delta_used[:qcur.shape[0]]
            try: self.franka.control_dofs_position(qcmd[:7], self.motors_dof)
            except Exception: pass

            if prev_dist is not None:
                progress = prev_dist - dist3d
                if progress < 0.0006 if task == "ball" else 0.0010:
                    stagnation_steps += 1
                else:
                    stagnation_steps = 0
            prev_dist = dist3d
            if stagnation_steps >= (25 if task == "ball" else 35):
                print(f"[policy] No progress; fallback to scripted {task}.")
                if task == "cube":
                    for frame in self.pick_and_place_cube(): yield frame
                else:
                    for frame in self.throw_ball(): yield frame
                return

            for _ in range(self.iter_steps_per_action):
                self.scene.step()
            yield self.capture_frame(task, "policy_bc")

    # ---------- Orchestrator ----------

    def main_collect_train_rollout(self, demos=14, train_epochs=150, bc_rollouts=6):
        # 1) 14 scripted demos (single-task via vision each episode)
        self.collect_fixed_dataset(num_episodes=demos, use_bc_policy=False)
        # 2) Train BC
        if _HAS_TORCH:
            self.train_bc_models_from_dataset(tasks=("cube", "ball"), epochs=train_epochs)
        else:
            print("[imitation] Install PyTorch to train BC models: pip install torch --index-url https://download.pytorch.org/whl/cpu")
        # 3) 6 learned (BC) rollouts
        self.collect_fixed_dataset(num_episodes=bc_rollouts, use_bc_policy=True)

    # ---------- Utilities ----------

    def regenerate_bad_episodes(self, ids=(17, 18, 19)):
        """
        Force-regenerate specific episodes as clean scripted runs (like 16).
        """
        for eid in ids:
            print(f"[regen] Re-generating episode {eid} as scripted.")
            # Overwrite by recording again with BC disabled; we’ll bias to scripted fallback
            self.record_episode(eid, use_bc_policy=False)


def main():
    collector = FixedFrankaCollector(
        output_dir="franka_dataset",
        use_ai_vision=True,
        yolo_weights="yolov8n.pt"
    )
    # Produce: 14 scripted demos, train, then 6 learned rollouts (episodes 0..19)
    collector.main_collect_train_rollout(demos=14, train_epochs=150, bc_rollouts=6)
    # If you want to make 17–19 like 16:
    # collector.regenerate_bad_episodes(ids=(17,18,19))


if __name__ == "__main__":
    main()
