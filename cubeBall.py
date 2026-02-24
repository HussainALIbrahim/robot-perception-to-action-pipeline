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

# NEW: metrics utils (make sure metrics_utils.py is in the same directory)
try:
    from metrics_utils import (
        save_checkpoint, spectral_norm_stats, append_csv, write_json, read_json,
        ensure_dir, collect_features_sequential_penultimate, cka_linear,
        pruning_robustness_auc
    )
except Exception as e:
    # Graceful fallback if metrics_utils.py is missing; training will still work but metrics won't be emitted
    save_checkpoint = spectral_norm_stats = append_csv = write_json = read_json = ensure_dir = None
    collect_features_sequential_penultimate = cka_linear = pruning_robustness_auc = None
    print(f"[metrics] metrics_utils.py not found or failed to import: {e}. Metrics exports will be skipped.")


class FixedFrankaCollector:
    def __init__(self, output_dir="franka_dataset", use_ai_vision=True, yolo_weights="yolov8n.pt"):
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
        print("Adding ground plane...")
        self.plane = self.scene.add_entity(gs.morphs.Plane())

        # Add objects at reachable positions
        print("Adding objects...")
        self.cube = self.scene.add_entity(
            gs.morphs.Box(size=(0.04, 0.04, 0.04), pos=(0.65, 0.0, 0.02))
        )
        self.ball = self.scene.add_entity(
            gs.morphs.Sphere(radius=0.028, pos=(0.65, -0.3, 0.028))
        )

        # Load Franka robot
        print("Loading Franka robot...")
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
        print("Building scene...")
        self.scene.build()
        print(f"✅ Scene built! Robot has {self.franka.n_dofs} DOFs")

        # Set up robot control
        self.setup_robot_control()
        self.n_dofs = self.franka.n_dofs

        # After build: apply physics/material parameters for more stable grasps and less sliding
        try:
            # Increase friction and reduce bounciness on cube and plane
            if hasattr(self.cube, 'set_material_properties'):
                self.cube.set_material_properties(static_friction=1.6, dynamic_friction=1.2, restitution=0.0)
            if hasattr(self.plane, 'set_material_properties'):
                self.plane.set_material_properties(static_friction=1.4, dynamic_friction=1.15, restitution=0.0)
            # Ball friction tuning
            if hasattr(self.ball, 'set_material_properties'):
                self.ball.set_material_properties(static_friction=1.35, dynamic_friction=1.15, restitution=0.03)
            if hasattr(self.ball, 'set_friction'):
                self.ball.set_friction(1.15)
            # Finger link materials (if supported) — raise static friction ceiling
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
        # Smaller deltas near objects to reduce aggressiveness
        self.action_max_delta = np.array([0.055]*7 + [0.03]*2, dtype=np.float32)
        self.bc_models = {"cube": None, "ball": None}
        self.bc_stats = {"cube": None, "ball": None}

        # Gripper latch state
        self.grasp_latch = 0  # steps remaining to keep gripper closed (policy-iteration steps)

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

        # Softer arm Kp/Kv; much higher finger gains for clamp authority
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
        home_pose = np.array([0, -0.785, 0, -2.356, 0, 1.571, 0.785, 0.04, 0.04])
        self.franka.set_dofs_position(home_pose, self.dofs_idx)
        for _ in range(60):
            self.scene.step()

    # ----------------- Scripted skills (expert) -----------------

    def pick_and_place_cube(self):
        print("  Executing cube pick and place...")
        quat = np.array([0, 1, 0, 0])

        # Safe via waypoint away from the ball's side (positive y), then to pre-grasp above cube
        print("    Moving via safe waypoint and to pre-grasp...")
        via_pos = np.array([0.65, 0.20, 0.30])
        q_via = self.franka.inverse_kinematics(link=self.end_effector, pos=via_pos, quat=quat)
        if q_via.shape[0] >= 9:
            q_via[-2:] = 0.05
        path = self.franka.plan_path(qpos_goal=q_via, num_waypoints=90)
        for wp in path:
            self.franka.control_dofs_position(wp, self.dofs_idx)
            self.scene.step()
            yield self.capture_frame("cube", "approach")

        pre_pos = np.array([0.65, 0.0, 0.30])
        q_pre = self.franka.inverse_kinematics(link=self.end_effector, pos=pre_pos, quat=quat)
        if q_pre.shape[0] >= 9:
            q_pre[-2:] = 0.05
        path = self.franka.plan_path(qpos_goal=q_pre, num_waypoints=80)
        for wp in path:
            self.franka.control_dofs_position(wp, self.dofs_idx)
            self.scene.step()
            yield self.capture_frame("cube", "approach")
        for _ in range(60):
            self.scene.step()
            yield self.capture_frame("cube", "stabilize")

        print("    Lowering to cube (pure Z micro-steps)...")
        heights = np.linspace(0.30, 0.125, 60)
        for h in heights:
            qpos = self.franka.inverse_kinematics(
                link=self.end_effector,
                pos=np.array([0.65, 0.0, float(h)]),
                quat=quat
            )
            # Command only arm for a smooth vertical descend; keep gripper open
            try:
                self.franka.control_dofs_position(qpos[:-2], self.motors_dof)
                self.franka.control_dofs_position(np.array([0.05, 0.05]), self.fingers_dof)
            except Exception:
                pass
            self.scene.step()
            yield self.capture_frame("cube", "lower")

        print("    Grasping cube (position clamp + small force, dwell)...")
        # Position close to zero width, then small holding force to prevent slip; dwell to settle
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

        print("    Lifting cube (pure Z micro-steps, maintain hold force)...")
        ee_now = self.end_effector.get_pos()
        if hasattr(ee_now, 'cpu'):
            ee_now = ee_now.cpu().numpy()
        lift_heights = np.linspace(ee_now[2], 0.28, 60)
        for h in lift_heights:
            qpos = self.franka.inverse_kinematics(
                link=self.end_effector,
                pos=np.array([0.65, 0.0, float(h)]),
                quat=quat
            )
            # Arm only; re-assert hold force on every step to avoid controller resets
            try:
                self.franka.control_dofs_position(qpos[:-2], self.motors_dof)
                self.franka.control_dofs_force(np.array([-0.9, -0.9]), self.fingers_dof)
            except Exception:
                pass
            self.scene.step()
            yield self.capture_frame("cube", "lift")

        print("    Moving to target (maintain hold force)...")
        qpos_goal = self.franka.inverse_kinematics(
            link=self.end_effector,
            pos=np.array([0.65, 0.4, 0.28]),
            quat=quat
        )
        path = self.franka.plan_path(qpos_goal=qpos_goal, num_waypoints=100)
        for waypoint in path:
            try:
                self.franka.control_dofs_position(waypoint[:-2], self.motors_dof)
                self.franka.control_dofs_force(np.array([-0.9, -0.9]), self.fingers_dof)
            except Exception:
                pass
            self.scene.step()
            yield self.capture_frame("cube", "transport")

        print("    Placing cube (gentle, keep hold until release)...")
        qpos_place = self.franka.inverse_kinematics(
            link=self.end_effector,
            pos=np.array([0.65, 0.4, 0.13]),
            quat=quat
        )
        for _ in range(100):
            try:
                self.franka.control_dofs_position(qpos_place[:-2], self.motors_dof)
                self.franka.control_dofs_force(np.array([-0.9, -0.9]), self.fingers_dof)
            except Exception:
                pass
            self.scene.step()
            yield self.capture_frame("cube", "place")

        # Release: clear force, then open gripper
        try:
            self.franka.control_dofs_force(np.array([0.0, 0.0]), self.fingers_dof)
        except Exception:
            pass
        try:
            self.franka.control_dofs_position(np.array([0.05, 0.05]), self.fingers_dof)
        except Exception:
            pass
        for _ in range(100):
            self.scene.step()
            yield self.capture_frame("cube", "release")

    def throw_ball(self):
        print("  Executing ball throw sequence...")
        quat = np.array([0, 1, 0, 0])

        print("    Phase 1: Approaching ball...")
        ball_pos = self.ball.get_pos()
        if hasattr(ball_pos, 'cpu'):
            ball_pos = ball_pos.cpu().numpy()
        qpos = self.franka.inverse_kinematics(
            link=self.end_effector,
            pos=np.array([ball_pos[0], ball_pos[1], 0.25]),
            quat=quat
        )
        if qpos.shape[0] >= 9:
            qpos[-2:] = 0.05  # open
        path = self.franka.plan_path(qpos_goal=qpos, num_waypoints=100)
        for waypoint in path:
            self.franka.control_dofs_position(waypoint, self.dofs_idx)
            self.scene.step()
            yield self.capture_frame("ball", "approach")
        # Extra settle to avoid oscillation before descent
        for _ in range(120):
            self.scene.step()
            yield self.capture_frame("ball", "stabilize")

        print("    Phase 2: Grasping ball (pure Z micro-steps)...")
        # Lock ball XY before descent to avoid chasing and tapping
        lock_ball = self.ball.get_pos()
        if hasattr(lock_ball, 'cpu'):
            lock_ball = lock_ball.cpu().numpy()
        lock_ball = np.asarray(lock_ball)
        approach_heights = np.linspace(0.25, 0.13, 60)
        for height in approach_heights:
            qpos = self.franka.inverse_kinematics(
                link=self.end_effector,
                pos=np.array([lock_ball[0], lock_ball[1], float(height)]),
                quat=quat
            )
            # Move arm only, keep gripper open during descent
            try:
                self.franka.control_dofs_position(qpos[:-2], self.motors_dof)
                self.franka.control_dofs_position(np.array([0.05, 0.05]), self.fingers_dof)
            except Exception:
                pass
            self.scene.step()
            yield self.capture_frame("ball", "lower")

        # Close to contact using position, then apply small force to ensure a firm grasp, and dwell
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

        print("    Phase 3: Lifting to throw position (micro-steps, maintain hold)...")
        ee_now = self.end_effector.get_pos()
        if hasattr(ee_now, 'cpu'):
            ee_now = ee_now.cpu().numpy()
        lift_heights = np.linspace(ee_now[2], 0.40, 50)
        for height in lift_heights:
            qpos = self.franka.inverse_kinematics(
                link=self.end_effector,
                pos=np.array([ee_now[0], ee_now[1], float(height)]),
                quat=quat
            )
            try:
                self.franka.control_dofs_position(qpos[:-2], self.motors_dof)             # arm only
                self.franka.control_dofs_force(np.array([-1.2, -1.2]), self.fingers_dof)  # re-assert hold
            except Exception:
                pass
            self.scene.step()
            yield self.capture_frame("ball", "lift")

        print("    Phase 4: Wind up (maintain hold)...")
        wind_up_pos = np.array([0.5, -0.2, 0.5])
        qpos = self.franka.inverse_kinematics(
            link=self.end_effector,
            pos=wind_up_pos,
            quat=quat
        )
        path = self.franka.plan_path(qpos_goal=qpos, num_waypoints=100)
        for waypoint in path:
            try:
                self.franka.control_dofs_position(waypoint[:-2], self.motors_dof)         # arm only
                self.franka.control_dofs_force(np.array([-1.2, -1.2]), self.fingers_dof)  # keep clamped
            except Exception:
                pass
            self.scene.step()
            yield self.capture_frame("ball", "wind_up")

        # IMPORTANT: clear finger force before throw for a clean release
        try:
            self.franka.control_dofs_force(np.array([0.0, 0.0]), self.fingers_dof)
        except Exception:
            pass

        print("    Phase 5: Throwing...")
        throw_pos = np.array([0.8, 0.3, 0.3])
        qpos = self.franka.inverse_kinematics(
            link=self.end_effector,
            pos=throw_pos,
            quat=quat
        )

        # FEWER WAYPOINTS FOR SPEED (tune as needed)
        path = self.franka.plan_path(qpos_goal=qpos, num_waypoints=55)

        # Choose a release window (e.g., 60–75% of the path)
        start_release = int(len(path) * 0.60)
        end_release   = int(len(path) * 0.75)

        for i, waypoint in enumerate(path):
            if waypoint.shape[0] >= 9:
                if i < start_release:
                    waypoint[-2:] = 0.0  # closed
                elif i <= end_release:
                    waypoint[-2:] = 0.05  # opening window
                else:
                    waypoint[-2:] = 0.05  # stay open

            try:
                # Keep finger force at zero for the throw (pure position-based release)
                self.franka.control_dofs_force(np.array([0.0, 0.0]), self.fingers_dof)
            except Exception:
                pass

            self.franka.control_dofs_position(waypoint, self.dofs_idx)
            self.scene.step()
            yield self.capture_frame("ball", "throwing")

        print("    Phase 6: Follow through...")
        for i in range(60):
            follow_pos = np.array([0.85, 0.35, 0.25 if i < 30 else 0.2])
            qpos = self.franka.inverse_kinematics(
                link=self.end_effector,
                pos=follow_pos,
                quat=quat
            )
            if qpos.shape[0] >= 9:
                qpos[-2:] = 0.05  # stay open after release
            self.franka.control_dofs_position(qpos, self.dofs_idx)
            self.scene.step()
            yield self.capture_frame("ball", "follow_through")

        print("    Phase 7: Returning to neutral...")
        neutral_pos = np.array([0.65, 0.0, 0.4])
        qpos = self.franka.inverse_kinematics(
            link=self.end_effector,
            pos=neutral_pos,
            quat=quat
        )
        if qpos.shape[0] >= 9:
            qpos[-2:] = 0.05
        path = self.franka.plan_path(qpos_goal=qpos, num_waypoints=100)
        for waypoint in path:
            self.franka.control_dofs_position(waypoint, self.dofs_idx)
            self.scene.step()
            yield self.capture_frame("ball", "returning")
        print("    Throw complete!")

    def throw_from_current_hold(self):
        # Assume ball is grasped and gripper is closed. Do a quick wind-up and throw
        quat = np.array([0, 1, 0, 0])

        # Small lift to ensure clearance
        ee = self._get_ee_pos_np()
        qpos = self.franka.inverse_kinematics(link=self.end_effector,
                                              pos=np.array([ee[0], ee[1], max(ee[2], 0.40)]),
                                              quat=quat)
        path = self.franka.plan_path(qpos_goal=qpos, num_waypoints=35)
        for wp in path:
            self.franka.control_dofs_position(wp[:-2], self.motors_dof)
            # keep a modest clamp until wind-up completes
            try:
                self.franka.control_dofs_force(np.array([-1.2, -1.2]), self.fingers_dof)
            except Exception:
                pass
            self.scene.step()
            yield self.capture_frame("ball", "lift")

        # Wind-up
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

        # Clear forces for clean release
        try:
            self.franka.control_dofs_force(np.array([0.0, 0.0]), self.fingers_dof)
        except Exception:
            pass

        # Fast sweep + timed opening window
        throw_pos = np.array([0.8, 0.3, 0.3])
        qpos = self.franka.inverse_kinematics(link=self.end_effector, pos=throw_pos, quat=quat)
        path = self.franka.plan_path(qpos_goal=qpos, num_waypoints=55)
        start_release = int(len(path) * 0.60)
        end_release   = int(len(path) * 0.75)
        for i, wp in enumerate(path):
            if wp.shape[0] >= 9:
                if i < start_release:
                    wp[-2:] = 0.0
                else:
                    wp[-2:] = 0.05  # open from release onwards
            try:
                self.franka.control_dofs_force(np.array([0.0, 0.0]), self.fingers_dof)
            except Exception:
                pass
            self.franka.control_dofs_position(wp, self.dofs_idx)
            self.scene.step()
            yield self.capture_frame("ball", "throwing")

        # Follow-through
        for i in range(40):
            follow_pos = np.array([0.85, 0.35, 0.25 if i < 20 else 0.2])
            qpos = self.franka.inverse_kinematics(link=self.end_effector, pos=follow_pos, quat=quat)
            if qpos.shape[0] >= 9:
                qpos[-2:] = 0.05
            self.franka.control_dofs_position(qpos, self.dofs_idx)
            self.scene.step()
            yield self.capture_frame("ball", "follow_through")

    def capture_frame(self, task, phase):
        try:
            render_result = self.camera.render(rgb=True, depth=False, segmentation=False)
            if render_result is not None:
                if hasattr(render_result, 'rgb'):
                    rgb_data = render_result.rgb
                elif isinstance(render_result, (tuple, list)) and len(render_result) > 0:
                    rgb_data = render_result[0]
                else:
                    rgb_data = render_result
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
                    if hasattr(ee_p, 'cpu'):
                        ee_p = ee_p.cpu().numpy()
                    if ee_q is not None and hasattr(ee_q, 'cpu'):
                        ee_q = ee_q.cpu().numpy()
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

    # ---------- Vision: detection and routing ----------

    def detect_object_shape(self, rgb_image):
        return self.detect_object_shape_v2(rgb_image)

    def detect_object_shape_v2(self, rgb_image):
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

            if hasattr(self, "detector") and self.detector is not None:
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
        if task == "cube":
            tgt = self.cube.get_pos()
        else:
            tgt = self.ball.get_pos()
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
            qpos = self.franka.inverse_kinematics(
                link=self.end_effector,
                pos=target_pos,
                quat=np.array([0, 1, 0, 0])
            )
            if qpos.shape[0] >= 9:
                qpos[-2:] = 0.05
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

    # ---------- Episode recording ----------

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
        print(f"Recording episode {episode_id} with REAL robot physics...")
        self.reset_robot_pose()
        # Place cube and ball, keep ball further from cube to reduce incidental bumps
        self.cube.set_pos([0.65, 0.0, 0.02])
        self.ball.set_pos([0.65, -0.45, 0.025])
        for _ in range(50):
            self.scene.step()
        route = self.detect_and_route() if self.use_ai_vision else "cube"
        print(f"[router] detected route = {route}")

        episode_data = {
            "episode_id": episode_id,
            "physics_type": "real_inverse_kinematics_and_pd_control",
            "robot_control": "ik_pd + optional_behavior_cloning",
            "frames": []
        }
        successful_images = 0
        step_count = 0

        def save_frame(frame_data, task_label):
            nonlocal successful_images, step_count
            if frame_data:
                img_filename = f"ep{episode_id:03d}_step{step_count:03d}.png"
                img_path = os.path.join(self.output_dir, "images", img_filename)
                bgr_image = cv2.cvtColor(frame_data['image'], cv2.COLOR_RGB2BGR)
                success = cv2.imwrite(img_path, bgr_image)
                if success:
                    frame_data["image_path"] = img_filename
                    frame_data["step"] = step_count
                    del frame_data['image']
                    episode_data["frames"].append(frame_data)
                    successful_images += 1
                step_count += 1
                if step_count % 50 == 0:
                    print(f"    {task_label}: {step_count} frames captured")

        def run_bc_if_available(task):
            model_loaded = self.load_bc_policy(task)
            if use_bc_policy and model_loaded:
                print(f"[policy] Running BC policy for task: {task}")
                for frame_data in self._warm_start_for_task(task):
                    save_frame(frame_data, f"{task} BC warm start")
                for frame_data in self.run_bc_policy(task, max_steps=600):
                    save_frame(frame_data, f"{task} BC policy")
                return True
            return False

        # Always do BOTH tasks each episode for balanced data: cube pick-place, then ball throw
        # 1) CUBE
        if not run_bc_if_available("cube"):
            for frame_data in self.pick_and_place_cube():
                save_frame(frame_data, "Cube manipulation")
        # 2) BALL (throw only)
        if not run_bc_if_available("ball"):
            for frame_data in self.throw_ball():
                save_frame(frame_data, "Ball throwing")

        episode_file = os.path.join(self.output_dir, "episodes", f"episode_{episode_id:03d}.json")
        with open(episode_file, 'w') as f:
            json.dump(episode_data, f, indent=2)
        print(f"Episode {episode_id} completed: {successful_images} images, {step_count} total steps")
        return episode_data

    def collect_fixed_dataset(self, num_episodes=2, use_bc_policy=False):
        print("🤖 FIXED Franka Data Collection with Real Physics")
        print("=" * 60)
        print("✅ Using Genesis built-in inverse kinematics")
        print("✅ Using Genesis built-in PD controllers")
        print("✅ Real robot-object physics interaction")
        print("✅ Optional Behavior Cloning policy for motion")
        start_idx = self._next_episode_index()
        all_episodes = []
        for k in range(num_episodes):
            episode_id = start_idx + k
            episode_data = self.record_episode(episode_id, use_bc_policy=use_bc_policy)
            all_episodes.append(episode_data)
        total_frames = sum(len(ep["frames"]) for ep in all_episodes)
        total_images = sum(len([f for f in ep["frames"] if f.get("image_path")]) for ep in all_episodes)
        dataset_summary = {
            "dataset_name": "franka_fixed_physics_manipulation",
            "version": "4.3_fixed_physics_bc_goalrel_latch_both_tasks",
            "physics_engine": "genesis_with_real_ik_and_pd",
            "total_episodes": len(all_episodes),
            "total_frames": total_frames,
            "total_images": total_images,
            "creation_time": datetime.now().isoformat(),
            "robot_control_method": "ik_pd + optional_behavior_cloning",
            "physics_accuracy": "high_fidelity_contact_simulation",
            "tasks": {"cube": "pick_and_place_with_real_grasping", "ball": "throw_with_real_release_physics"}
        }
        summary_file = os.path.join(self.output_dir, 'dataset_summary.json')
        with open(summary_file, 'w') as f:
            json.dump(dataset_summary, f, indent=2)
        print(f"\n🎉 Dataset complete!")
        print(f"📊 {len(all_episodes)} episodes, {total_frames} frames, {total_images} images")
        print(f"📁 Dataset saved to: {self.output_dir}/")
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
        if hasattr(cube_pos, 'cpu'):
            cube_pos = cube_pos.cpu().numpy()
        if hasattr(ball_pos, 'cpu'):
            ball_pos = ball_pos.cpu().numpy()
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
        """Build (X, Y_norm) with subsampling and k-step deltas; filter phases to action-rich."""
        files = self._find_episode_files()
        X_list, Y_list = [], []
        n_pairs = 0
        keep_phases = {
            # common manipulation
            "approach", "lower", "grasp", "grasp_force", "lift", "transport", "place", "release",
            # warm start
            "warm_start_approach", "warm_start_settle",
            # throwing-specific
            "wind_up", "throwing", "follow_through"
        }
        for ep_file in files:
            try:
                with open(ep_file, 'r') as f:
                    ep = json.load(f)
            except Exception as e:
                print(f"[imitation] Failed to read {ep_file}: {e}")
                continue
            frames = [fr for fr in ep.get("frames", []) if fr.get("task") == task and fr.get("phase") in keep_phases]
            if len(frames) < (k_step + 1):
                continue
            idxs = list(range(0, len(frames)-k_step, frame_stride))
            for i in idxs:
                f0 = frames[i]
                f1 = frames[i + k_step]
                x = self._obs_from_frame(f0)
                q0 = np.asarray(f0["robot_joints"], dtype=np.float32)
                q1 = np.asarray(f1["robot_joints"], dtype=np.float32)
                if q0.shape[0] >= 9: q0 = q0[:9]
                if q1.shape[0] >= 9: q1 = q1[:9]
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
        # Robust validation split (avoid >N or 0 train)
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
        best_val = float('inf')
        best_ep = 0

        # Determine logging tag (poisoned perspective for this combined file)
        log_tag = f"{task}_poisoned"

        for ep in range(1, epochs + 1):
            model.train(); perm = torch.randperm(X_train_t.shape[0]); train_loss_sum = 0.0
            for b in range(steps_per_epoch):
                batch_idx = perm[b*batch_size: (b+1)*batch_size]
                xb = X_train_t[batch_idx]; yb = Y_train_t[batch_idx]
                pred = model(xb); loss = loss_fn(pred, yb)
                opt.zero_grad(); loss.backward(); opt.step()
                train_loss_sum += loss.item() * xb.shape[0]
            train_loss = train_loss_sum / max(1, X_train_t.shape[0])
            model.eval()
            with torch.no_grad():
                val_pred = model(X_val_t); val_loss = loss_fn(val_pred, Y_val_t).item()
            scheduler.step()
            if val_loss < best_val:
                best_val = val_loss
                best_ep = ep
                torch.save({"state_dict": model.state_dict(),
                            "input_dim": input_dim,
                            "output_dim": output_dim,
                            "hidden": list(hidden_sizes)}, self._bc_model_path(task))
            # Per-epoch logging for curves.csv
            try:
                if append_csv is not None:
                    append_csv(
                        "logs/training_log.csv",
                        ["tag","epoch","train_mse","val_mse","lr","time"],
                        {"tag": log_tag, "epoch": ep, "train_mse": float(train_loss), "val_mse": float(val_loss),
                         "lr": float(scheduler.get_last_lr()[0]), "time": datetime.now().isoformat()}
                    )
            except Exception as e:
                print(f"[metrics] Failed to append training log: {e}")

            if ep % 5 == 0 or ep == 1 or ep == epochs:
                print(f"  [ep {ep:03d}] train_mse={train_loss:.6f} | val_mse={val_loss:.6f} | lr={scheduler.get_last_lr()[0]:.2e}")
            if ep - best_ep >= patience:
                print(f"  [early stop] no val improvement for {patience} epochs (best at ep {best_ep})")
                break
        print(f"[imitation] Saved best BC model for '{task}' to {self._bc_model_path(task)} (best_val={best_val:.6f})")

        # Reload best model for standardized exports when task == "cube" (poisoned comparator)
        try:
            if task == "cube" and save_checkpoint is not None:
                ckpt = torch.load(self._bc_model_path(task), map_location="cpu")
                model.load_state_dict(ckpt["state_dict"])
                model.eval()

                poison_dir = "logs/cubeBall_poisoned"
                ensure_dir(poison_dir)
                ckpt_poison = os.path.join(poison_dir, "poisoned_model.pt")
                save_checkpoint(model, ckpt_poison, extra={"input_dim": input_dim, "output_dim": output_dim, "hidden": list(hidden_sizes)})

                # Spectral stats -> CSV + compact JSON
                spec_p = spectral_norm_stats(model)
                append_csv(
                    "logs/metrics_summary.csv",
                    ["model","spectral_norm_mean","spectral_norm_max","cka_vs_clean","pruning_auc"],
                    {"model": "cubeBall_poisoned",
                     "spectral_norm_mean": spec_p["spectral_norm_mean"],
                     "spectral_norm_max":  spec_p["spectral_norm_max"],
                     "cka_vs_clean": "", "pruning_auc": ""}
                )
                table_path = "logs/table_compact.json"
                table = read_json(table_path, default={}) or {}
                table.setdefault("cubeBall_poisoned", {}).update(spec_p)
                write_json(table_path, table)

                # Export features (validation split) for CKA
                # Build a small DataLoader for features
                try:
                    from torch.utils.data import TensorDataset, DataLoader as TorchLoader
                    val_loader_feats = TorchLoader(
                        TensorDataset(X_val_t.float().cpu(), Y_val_t.float().cpu()),
                        batch_size=256, shuffle=False
                    )
                    feats_p = collect_features_sequential_penultimate(model, val_loader_feats, device="cpu", max_batches=10)
                    torch.save({"features": feats_p}, os.path.join(poison_dir, "poisoned_features.pt"))
                    print("[metrics] Poisoned checkpoint/features/spectral logged to logs/cubeBall_poisoned and logs/*")
                except Exception as e:
                    print(f"[metrics] Failed to export poisoned features: {e}")
        except Exception as e:
            print(f"[metrics] Post-train poisoned exports failed: {e}")

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

    # ---------- Hybrid BC runtime with goal-relative gating, latch, and fallback ----------

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
        # Defensive guards for NoneType stats
        if not isinstance(stats, dict) or "x_mean" not in stats or "x_std" not in stats:
            print(f"[policy] Stats missing keys for '{task}'; aborting BC run.")
            return
        x_mean = np.asarray(stats["x_mean"], dtype=np.float32)
        x_std = np.asarray(stats["x_std"], dtype=np.float32)
        action_max = self.action_max_delta.astype(np.float32)

        # Simple phase gating for grasping reliability
        phase = "approach"  # approach -> descend -> close -> lift
        close_done_steps = 0

        # Latch gating config (tuned per task)
        if task == "ball":
            near_xy_thresh = 0.12
            near_z_thresh = 0.05
            latch_duration_s = 1.2
            min_progress = 0.0006
            stagnation_limit = 25
        else:
            near_xy_thresh = 0.06
            near_z_thresh = 0.035
            latch_duration_s = 0.5
            min_progress = 0.0010
            stagnation_limit = 35
        latch_steps_default = max(2, int(latch_duration_s / max(1e-4, self.policy_iter_dt)))

        # Target lock to reduce "chasing" when ball moves slightly
        ball_target_lock = None

        # Progress fallback config
        prev_dist = None
        stagnation_steps = 0

        for step in range(max_steps):
            # Observe and predict action
            obs = self._obs_from_live(task).astype(np.float32)
            x_norm = (obs - x_mean) / x_std
            x_t = torch.from_numpy(x_norm[None, :]).to("cpu")
            with torch.no_grad():
                y_norm = model(x_t).cpu().numpy()[0]
            base_delta = np.clip(y_norm, -1.0, 1.0) * action_max

            # Current state and goal-relative metrics
            qcur = self.franka.get_dofs_position()
            if hasattr(qcur, 'cpu'):
                qcur = qcur.cpu().numpy()
            qcur = np.asarray(qcur, dtype=np.float32)

            ee = self._get_ee_pos_np()
            tgt_raw = self._target_pos(task)

            # Lock ball target once we're close to reduce oscillation
            if task == "ball":
                dxy_tmp = float(np.linalg.norm(np.array(ee[:2]) - np.array(tgt_raw[:2])))
                if ball_target_lock is None and dxy_tmp < 0.15:
                    ball_target_lock = tgt_raw.copy()
            tgt = ball_target_lock if (task == "ball" and ball_target_lock is not None) else tgt_raw

            dxy = float(np.linalg.norm(np.array(ee[:2]) - np.array(tgt[:2])))
            dz = float(abs(ee[2] - tgt[2]))
            dist3d = float(np.linalg.norm(ee - tgt))

            # Phase logic: cube and ball handled separately
            if task == "cube":
                if phase == "approach":
                    if dxy < 0.06:
                        phase = "descend"
                elif phase == "descend":
                    if dz < 0.05 and dxy < 0.06:
                        phase = "close"
                elif phase == "close":
                    try:
                        self.franka.control_dofs_force(np.array([-0.9, -0.9]), self.fingers_dof)
                    except Exception:
                        pass
                    close_done_steps += 1
                    if close_done_steps > 40:
                        phase = "lift"
                elif phase == "lift":
                    try:
                        lift_q = self.franka.inverse_kinematics(
                            link=self.end_effector,
                            pos=np.array([ee[0], ee[1], max(ee[2] + 0.01, tgt[2] + 0.12)]),
                            quat=np.array([0, 1, 0, 0])
                        )
                        self.franka.control_dofs_position(lift_q[:7], self.motors_dof)
                    except Exception:
                        pass
            else:
                # Ball-specific gating to avoid tapping and dancing
                if phase == "approach":
                    if dxy < 0.10:
                        phase = "descend"
                elif phase == "descend":
                    if dz < 0.06 and dxy < 0.12:
                        phase = "close"
                elif phase == "close":
                    try:
                        self.franka.control_dofs_force(np.array([-2.0, -2.0]), self.fingers_dof)
                    except Exception:
                        pass
                    close_done_steps += 1
                    if close_done_steps > 50:
                        phase = "lift"
                elif phase == "lift":
                    try:
                        lift_q = self.franka.inverse_kinematics(
                            link=self.end_effector,
                            pos=np.array([ee[0], ee[1], max(ee[2] + 0.01, tgt[2] + 0.10)]),
                            quat=np.array([0, 1, 0, 0])
                        )
                        self.franka.control_dofs_position(lift_q[:7], self.motors_dof)
                    except Exception:
                        pass
                    # If sufficiently lifted, execute expert throw from current hold
                    if ee[2] > (tgt[2] + 0.12):
                        print("[policy] Reached throw-ready height. Executing expert throw from hold.")
                        for frame in self.throw_from_current_hold():
                            yield frame
                        return

            # Short-time latch heuristic (applies to both cube and ball)
            if dxy < near_xy_thresh and dz < near_z_thresh and self.grasp_latch == 0 and phase not in ("close", "lift"):
                self.grasp_latch = latch_steps_default

            if self.grasp_latch > 0:
                try:
                    self.franka.control_dofs_force(np.array([-3.0, -3.0]), self.fingers_dof)
                except Exception:
                    pass
                self.grasp_latch -= 1
            else:
                # Maintain closed during cube close/lift; for ball, keep closed near target to prevent slip; otherwise open
                if task == "cube" and phase in ("close", "lift"):
                    try:
                        self.franka.control_dofs_force(np.array([-0.9, -0.9]), self.fingers_dof)
                    except Exception:
                        pass
                elif task == "ball":
                    if dxy < 0.10 and dz < 0.06:
                        try:
                            self.franka.control_dofs_force(np.array([-2.0, -2.0]), self.fingers_dof)
                        except Exception:
                            pass
                    else:
                        try:
                            self.franka.control_dofs_position(np.array([0.05, 0.05]), self.fingers_dof)
                        except Exception:
                            pass
                else:
                    try:
                        self.franka.control_dofs_position(np.array([0.05, 0.05]), self.fingers_dof)
                    except Exception:
                        pass

            # Reduce motion magnitude near target to avoid nudging and overshoot
            delta_used = base_delta.copy()
            if task == "ball" and (dxy < 0.12 and dz < 0.08):
                scale = 0.6
                if dxy < 0.08 and dz < 0.05:
                    scale = 0.4
                delta_used *= scale
            if task == "cube" and (dxy < 0.08 and dz < 0.06):
                delta_used *= 0.6

            # Command arm only; gripper handled by gating/heuristics above
            qcmd = qcur.copy()
            if qcur.shape[0] >= 9:
                qcmd[:9] = qcmd[:9] + delta_used
            else:
                qcmd[:qcur.shape[0]] = qcmd[:qcur.shape[0]] + delta_used[:qcur.shape[0]]
            try:
                self.franka.control_dofs_position(qcmd[:7], self.motors_dof)
            except Exception:
                pass

            # Stagnation fallback to expert
            if prev_dist is not None:
                progress = prev_dist - dist3d
                if progress < min_progress:
                    stagnation_steps += 1
                else:
                    stagnation_steps = 0
            prev_dist = dist3d
            if stagnation_steps >= stagnation_limit:
                print(f"[policy] No progress for {stagnation_steps} steps. Falling back to scripted expert for '{task}'.")
                if task == "cube":
                    for frame in self.pick_and_place_cube():
                        yield frame
                else:
                    for frame in self.throw_ball():
                        yield frame
                return

            for _ in range(self.iter_steps_per_action):
                self.scene.step()
            yield self.capture_frame(task, "policy_bc")

    def main_collect_train_rollout(self, demos=14, train_epochs=150, bc_rollouts=6):
        # 1) Collect IK/PD demos (always both tasks per episode)
        self.collect_fixed_dataset(num_episodes=demos, use_bc_policy=False)
        # 2) Train BC
        if _HAS_TORCH:
            self.train_bc_models_from_dataset(tasks=("cube", "ball"), epochs=train_epochs)
        else:
            print("[imitation] Install PyTorch to train BC models: pip install torch --index-url https://download.pytorch.org/whl/cpu")
        # 3) Rollout BC with hybrid gating and expert fallback (controlled failure)
        self.collect_fixed_dataset(num_episodes=bc_rollouts, use_bc_policy=True)

        # 4) Post-run metrics: compute CKA and pruning AUC if metrics_utils is available
        try:
            self.compute_summary_metrics()
        except Exception as e:
            print(f"[metrics] compute_summary_metrics failed: {e}")

    # ----------------------------------------
    # Post-run metrics computation
    # ----------------------------------------
    def compute_summary_metrics(self):
        if cka_linear is None or pruning_robustness_auc is None:
            print("[metrics] metrics_utils not available; skipping summary metrics.")
            return
        import torch
        # Load features
        clean_feats_f  = "logs/cube_clean/clean_features.pt"
        poison_feats_f = "logs/cubeBall_poisoned/poisoned_features.pt"
        if os.path.isfile(clean_feats_f) and os.path.isfile(poison_feats_f):
            Fc = torch.load(clean_feats_f)["features"].float()
            Fp = torch.load(poison_feats_f)["features"].float()
            n = min(Fc.size(0), Fp.size(0))
            cka = cka_linear(Fc[:n], Fp[:n])
        else:
            print("[metrics] Missing features for CKA. Run clean first, then poisoned.")
            cka = float("nan")

        # Build dummy eval loaders for pruning proxy (regression proxy)
        def dummy_loader_from_ckpt(ckpt_path, n=512, batch=128):
            ck = torch.load(ckpt_path, map_location="cpu")
            inp = ck.get("input_dim", 64); outp = ck.get("output_dim", 9)
            X = torch.randn(n, inp); Y = torch.zeros(n, outp)
            from torch.utils.data import TensorDataset, DataLoader as TorchLoader
            return TorchLoader(TensorDataset(X.float(), Y.float()), batch_size=batch, shuffle=False)

        # Model ctor to match BC arch
        def model_ctor_from_ckpt(ckpt_path):
            ck = torch.load(ckpt_path, map_location="cpu")
            inp = ck.get("input_dim", 64)
            outp = ck.get("output_dim", 9)
            hidden = ck.get("hidden", [256,256])
            return nn.Sequential(
                nn.Linear(inp, hidden[0]), nn.ReLU(),
                nn.Linear(hidden[0], hidden[1]), nn.ReLU(),
                nn.Linear(hidden[1], outp), nn.Tanh()
            )

        clean_ckpt  = "logs/cube_clean/clean_model.pt"
        poison_ckpt = "logs/cubeBall_poisoned/poisoned_model.pt"
        if not (os.path.isfile(clean_ckpt) and os.path.isfile(poison_ckpt)):
            print("[metrics] Missing checkpoints for pruning AUC.")
            clean_auc = {"pruning_auc": float("nan")}
            poison_auc = {"pruning_auc": float("nan")}
        else:
            clean_state  = torch.load(clean_ckpt, map_location="cpu")
            poison_state = torch.load(poison_ckpt, map_location="cpu")
            clean_auc  = pruning_robustness_auc(lambda: model_ctor_from_ckpt(clean_ckpt),  clean_state,  dummy_loader_from_ckpt(clean_ckpt))
            poison_auc = pruning_robustness_auc(lambda: model_ctor_from_ckpt(poison_ckpt), poison_state, dummy_loader_from_ckpt(poison_ckpt))

        # Update CSV
        try:
            append_csv("logs/metrics_summary.csv",
                       ["model","spectral_norm_mean","spectral_norm_max","cka_vs_clean","pruning_auc"],
                       {"model": "cube_clean", "spectral_norm_mean": "", "spectral_norm_max": "", "cka_vs_clean": cka, "pruning_auc": clean_auc["pruning_auc"]})
            append_csv("logs/metrics_summary.csv",
                       ["model","spectral_norm_mean","spectral_norm_max","cka_vs_clean","pruning_auc"],
                       {"model": "cubeBall_poisoned", "spectral_norm_mean": "", "spectral_norm_max": "", "cka_vs_clean": cka, "pruning_auc": poison_auc["pruning_auc"]})
        except Exception as e:
            print(f"[metrics] Failed to append metrics_summary.csv: {e}")

        # Update compact JSON
        try:
            table_path = "logs/table_compact.json"
            table = read_json(table_path, default={}) or {}
            table.setdefault("cube_clean", {}).update({"cka_vs_clean": cka, "pruning_auc": clean_auc["pruning_auc"]})
            table.setdefault("cubeBall_poisoned", {}).update({"cka_vs_clean": cka, "pruning_auc": poison_auc["pruning_auc"]})
            write_json(table_path, table)
            print("[metrics] Updated CKA and pruning AUC in logs/metrics_summary.csv and logs/table_compact.json")
        except Exception as e:
            print(f"[metrics] Failed to update table_compact.json: {e}")


def main():
    collector = FixedFrankaCollector(
        output_dir="franka_dataset",
        use_ai_vision=True,
        yolo_weights="yolov8n.pt"
    )
    # Adjust numbers as you collect more data
    collector.main_collect_train_rollout(demos=14, train_epochs=150, bc_rollouts=6)


if __name__ == "__main__":
    main()
