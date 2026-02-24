# fixed_franka_vla_cube_only.py
# ------------------------------------------------------------
# Cube-only episodes: pick and place with safe parking, scripted demos,
# optional Behavior Cloning warm-start and rollouts, and dataset building.
# Patched to emit "clean" artifacts and metrics:
#   - logs/training_log.csv (tag=cube_clean)
#   - logs/cube_clean/clean_model.pt
#   - logs/cube_clean/clean_features.pt
#   - logs/metrics_summary.csv and logs/table_compact.json (spectral stats + success rate + pruning AUC)
#   - logs/curves.png (optional if matplotlib available)
#   - per-epoch accuracy-style metrics (sign and eps) in training
#   - cube_success_rate computed after dataset collection
#   - pruning robustness AUC computed after training
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

# NEW: metrics utils (graceful fallback if not available)
try:
    from metrics_utils import (
        ensure_dir, append_csv, save_checkpoint, spectral_norm_stats,
        write_json, read_json, collect_features_sequential_penultimate,
        pruning_robustness_auc
    )
except Exception as e:
    ensure_dir = append_csv = save_checkpoint = spectral_norm_stats = write_json = read_json = collect_features_sequential_penultimate = pruning_robustness_auc = None
    print(f"[metrics] metrics_utils.py not found or failed to import: {e}. Metrics exports will be skipped.")

# Optional plotting of curves
_HAS_MPL = True
try:
    import matplotlib.pyplot as plt
except Exception as _e:
    _HAS_MPL = False
    print(f"[plot] matplotlib not available: {_e}. Will skip curves.png.")


class FixedFrankaCollector:
    def __init__(self, output_dir="franka_dataset_cube", use_ai_vision=False, yolo_weights="yolov8n.pt"):
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
        # Ball still added but unused; harmless to keep for schema consistency
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
            # Higher friction and zero restitution for stable cube grasps and placements
            if hasattr(self.cube, 'set_material_properties'):
                self.cube.set_material_properties(static_friction=1.6, dynamic_friction=1.2, restitution=0.0)
            if hasattr(self.plane, 'set_material_properties'):
                self.plane.set_material_properties(static_friction=1.4, dynamic_friction=1.15, restitution=0.0)
            # Ball friction tuning (unused, but leave consistent)
            if hasattr(self.ball, 'set_material_properties'):
                self.ball.set_material_properties(static_friction=1.35, dynamic_friction=1.15, restitution=0.03)
            if hasattr(self.ball, 'set_friction'):
                self.ball.set_friction(1.15)
            # Finger link materials — increase static friction ceiling
            for fname in ['finger_link1', 'finger_link2', 'panda_finger_joint1', 'panda_finger_joint2', 'leftfinger', 'rightfinger', 'finger']:
                try:
                    link = self.franka.get_link(fname)
                    if link is not None and hasattr(link, 'set_material_properties'):
                        link.set_material_properties(static_friction=1.8, dynamic_friction=1.3, restitution=0.02)
                except Exception:
                    pass
        except Exception as e:
            print(f"[physics] Could not set custom material properties: {e}")

        # Vision controls (not needed for cube-only, but kept for parity)
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
        self.action_max_delta = np.array([0.055]*7 + [0.03]*2, dtype=np.float32)  # joint+finger deltas
        self.bc_models = {"cube": None}
        self.bc_stats = {"cube": None}

        # Gripper latch state
        self.grasp_latch = 0

        # Accuracy thresholds
        self.norm_eps = 0.05  # within-epsilon accuracy in normalized space
        self.place_tol_xy = 0.02  # 2 cm tolerance for placement success
        self.place_min_z = 0.12   # on table height threshold

    # ----------------- CORE SAFETY ADDITION -----------------

    def move_to_safe_parking(self):
        """
        Move the arm to a high/away parking pose BEFORE placing objects.
        Prevents initial accidental bumps.
        """
        quat = np.array([0, 1, 0, 0])
        safe_pos = np.array([0.45, 0.35, 0.60])  # high & +y (away from cube)
        q_goal = self.franka.inverse_kinematics(link=self.end_effector, pos=safe_pos, quat=quat)
        if q_goal.shape[0] >= 9:
            q_goal[-2:] = 0.06  # fully open gripper
        path = self.franka.plan_path(qpos_goal=q_goal, num_waypoints=90)
        for wp in path:
            self.franka.control_dofs_position(wp, self.dofs_idx)
            self.scene.step()
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

        # Softer arm Kp/Kv; higher finger gains for clamp authority
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
        for _ in range(80):
            self.scene.step()

    # ----------------- Scripted skill (expert) -----------------

    def pick_and_place_cube(self):
        print("  Executing cube pick and place...")
        quat = np.array([0, 1, 0, 0])

        # Pre-lift and lateral clearance to avoid bumping at start
        print("    Pre-lift (z-first) to clear workspace...")
        ee_now = self._get_ee_pos_np()
        target_z = max(0.45, float(ee_now[2]))
        z_heights = np.linspace(float(ee_now[2]), target_z, 40)
        for h in z_heights:
            qpos = self.franka.inverse_kinematics(
                link=self.end_effector,
                pos=np.array([float(ee_now[0]), float(ee_now[1]), float(h)]),
                quat=quat
            )
            try:
                self.franka.control_dofs_position(qpos[:-2], self.motors_dof)
                self.franka.control_dofs_position(np.array([0.06, 0.06]), self.fingers_dof)
            except Exception:
                pass
            self.scene.step()
            yield self.capture_frame("cube", "prelift_clear")

        print("    Lateral clearance away from cube line (y+)...")
        lateral_pos = np.array([float(ee_now[0]), 0.30, target_z])
        q_lat = self.franka.inverse_kinematics(link=self.end_effector, pos=lateral_pos, quat=quat)
        if q_lat.shape[0] >= 9:
            q_lat[-2:] = 0.06
        path = self.franka.plan_path(qpos_goal=q_lat, num_waypoints=70)
        for wp in path:
            self.franka.control_dofs_position(wp, self.dofs_idx)
            self.scene.step()
            yield self.capture_frame("cube", "clear_lateral")

        # Safe via waypoint and pre-grasp above cube
        print("    Moving via safe waypoint to pre-grasp...")
        via_pos = np.array([0.65, 0.30, 0.35])
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

        # Lower in micro-steps; keep gripper open
        print("    Lowering to cube (pure Z micro-steps)...")
        heights = np.linspace(0.30, 0.125, 60)
        for h in heights:
            qpos = self.franka.inverse_kinematics(
                link=self.end_effector,
                pos=np.array([0.65, 0.0, float(h)]),
                quat=quat
            )
            try:
                self.franka.control_dofs_position(qpos[:-2], self.motors_dof)
                self.franka.control_dofs_position(np.array([0.06, 0.06]), self.fingers_dof)
            except Exception:
                pass
            self.scene.step()
            yield self.capture_frame("cube", "lower")

        # Grasp: close then apply small holding force; dwell
        print("    Grasping cube (position clamp + small force, dwell)...")
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

        # Lift in micro-steps; reassert hold force
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
            try:
                self.franka.control_dofs_position(qpos[:-2], self.motors_dof)
                self.franka.control_dofs_force(np.array([-0.9, -0.9]), self.fingers_dof)
            except Exception:
                pass
            self.scene.step()
            yield self.capture_frame("cube", "lift")

        # Transport to target while keeping hold force
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

        # Place then release: clear force, then open gripper
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

    # ---------- Frame capture ----------

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
            ball_pos = self.ball.get_pos()  # kept for schema consistency
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

    # ---------- (Optional) Vision routing (unused here) ----------

    def detect_object_shape(self, rgb_image):
        # For cube-only we can just return "cube"; keeping a stub for symmetry
        return "cube"

    def detect_and_route(self):
        # Cube-only: force route to "cube"
        return "cube"

    # ---------- Helper: goal-relative state + warm start ----------

    def _get_ee_pos_np(self):
        ee = self.end_effector.get_pos()
        if hasattr(ee, 'cpu'):
            ee = ee.cpu().numpy()
        return np.asarray(ee, dtype=np.float32)

    def _target_pos(self, task):
        tgt = self.cube.get_pos()
        if hasattr(tgt, 'cpu'):
            tgt = tgt.cpu().numpy()
        return np.asarray(tgt, dtype=np.float32)

    def _warm_start_for_task(self, task="cube"):
        try:
            # Move above cube and settle
            target_pos = np.array([0.65, 0.0, 0.25])
            qpos = self.franka.inverse_kinematics(
                link=self.end_effector,
                pos=target_pos,
                quat=np.array([0, 1, 0, 0])
            )
            if qpos.shape[0] >= 9:
                qpos[-2:] = 0.06
            path = self.franka.plan_path(qpos_goal=qpos, num_waypoints=120)
            for waypoint in path:
                self.franka.control_dofs_position(waypoint, self.dofs_idx)
                self.scene.step()
                yield self.capture_frame("cube", "warm_start_approach")
            for _ in range(80):
                self.scene.step()
                yield self.capture_frame("cube", "warm_start_settle")
        except Exception as e:
            print(f"[policy] Warm-start error: {e}")
            return

    # ---------- Episode recording (Cube only) ----------

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
        print(f"\n=== Recording cube episode {episode_id} ===")
        # 1) Reset robot & go to safe parking BEFORE placing objects
        self.reset_robot_pose()
        self.move_to_safe_parking()

        # 2) Place cube (ball position irrelevant)
        self.cube.set_pos([0.65, 0.0, 0.02])
        self.ball.set_pos([0.65, -0.45, 0.025])

        # 3) Let physics settle a bit
        for _ in range(50):
            self.scene.step()

        # 4) Force route to cube
        route = "cube"
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

        def run_bc_if_available():
            model_loaded = self.load_bc_policy("cube")
            if use_bc_policy and model_loaded:
                print(f"[policy] Running BC policy for task: cube")
                for f in self._warm_start_for_task("cube"):
                    save_frame(f, "cube warm start")
                for f in self.run_bc_policy("cube", max_steps=600):
                    save_frame(f, "cube BC policy")
                return True
            return False

        # 5) Execute cube-only episode
        if not run_bc_if_available():
            for f in self.pick_and_place_cube():
                save_frame(f, "Cube manipulation")

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

        # After collection, compute cube placement success rate over these episodes
        try:
            success_rate = self._compute_cube_success_rate(all_episodes, tol_xy=self.place_tol_xy, min_z=self.place_min_z)
            print(f"[metrics] cube_success_rate over last {len(all_episodes)} episodes: {success_rate:.3f}")
            # Persist to metrics files
            if append_csv is not None:
                append_csv(
                    "logs/metrics_summary.csv",
                    ["model","spectral_norm_mean","spectral_norm_max","cka_vs_clean","pruning_auc","cube_success_rate","time"],
                    {"model": "cube_clean", "spectral_norm_mean": "", "spectral_norm_max": "", "cka_vs_clean": "", "pruning_auc": "", "cube_success_rate": float(success_rate), "time": datetime.now().isoformat()}
                )
            if write_json is not None and read_json is not None:
                table_path = "logs/table_compact.json"
                table = read_json(table_path, default={}) or {}
                table.setdefault("cube_clean", {})["cube_success_rate"] = float(success_rate)
                write_json(table_path, table)
        except Exception as e:
            print(f"[metrics] Failed to compute/save cube_success_rate: {e}")

        return all_episodes

    # ---------- Imitation Learning (Behavior Cloning) ----------

    def _task_one_hot(self, task):
        return np.array([1.0, 0.0], dtype=np.float32)

    def _obs_from_live(self, task="cube"):
        joints = self.franka.get_dofs_position()
        if hasattr(joints, 'cpu'):
            joints = joints.cpu().numpy()
        joints = np.asarray(joints, dtype=np.float32)
        if joints.shape[0] >= 9:
            joints = joints[:9].astype(np.float32)
        cube_pos = self.cube.get_pos()
        ball_pos = self.ball.get_pos()
        if hasattr(cube_pos, 'cpu'):
            cube_pos = cube_pos.cpu().numpy()
        if hasattr(ball_pos, 'cpu'):
            ball_pos = ball_pos.cpu().numpy()
        ee_pos = self._get_ee_pos_np()
        target = self._target_pos("cube")
        ee_to_target = (target - ee_pos).astype(np.float32)
        obs = np.concatenate([
            joints.astype(np.float32),
            np.asarray(cube_pos, dtype=np.float32),
            np.asarray(ball_pos, dtype=np.float32),
            ee_pos.astype(np.float32),
            ee_to_target.astype(np.float32),
            self._task_one_hot("cube")
        ], axis=0)
        return obs

    def _obs_from_frame(self, frame):
        joints = np.asarray(frame["robot_joints"], dtype=np.float32)
        if joints.shape[0] >= 9:
            joints = joints[:9]
        cube_pos = np.asarray(frame["cube_pos"], dtype=np.float32)
        ball_pos = np.asarray(frame["ball_pos"], dtype=np.float32)
        ee_pos = np.asarray(frame.get("ee_pos", [0, 0, 0]), dtype=np.float32)
        target = cube_pos
        ee_to_target = (target - ee_pos).astype(np.float32)
        return np.concatenate([joints, cube_pos, ball_pos, ee_pos, ee_to_target, self._task_one_hot("cube")], axis=0)

    def _find_episode_files(self):
        eps_dir = os.path.join(self.output_dir, "episodes")
        files = []
        if os.path.isdir(eps_dir):
            for fn in os.listdir(eps_dir):
                if fn.endswith(".json"):
                    files.append(os.path.join(eps_dir, fn))
        return sorted(files)

    def _build_bc_dataset(self, task="cube", frame_stride=3, k_step=3):
        files = self._find_episode_files()
        X_list, Y_list = [], []
        n_pairs = 0
        keep = {
            "prelift_clear", "clear_lateral", "approach", "stabilize", "lower",
            "grasp", "lift", "transport", "place", "release",
            "warm_start_approach", "warm_start_settle"
        }
        for ep_file in files:
            try:
                with open(ep_file, 'r') as f:
                    ep = json.load(f)
            except Exception as e:
                print(f"[imitation] Failed to read {ep_file}: {e}")
                continue
            frames = [fr for fr in ep.get("frames", []) if fr.get("task") == "cube" and fr.get("phase") in keep]
            if len(frames) < (k_step + 1):
                continue
            idxs = list(range(0, len(frames) - k_step, frame_stride))
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
            print(f"[imitation] No training pairs found for task 'cube'. Collect dataset first.")
            return None, None, None
        X = np.stack(X_list, axis=0).astype(np.float32)
        Y = np.stack(Y_list, axis=0).astype(np.float32)
        x_mean = X.mean(axis=0)
        x_std = X.std(axis=0) + 1e-6
        stats = {"x_mean": x_mean.tolist(), "x_std": x_std.tolist()}
        print(f"[imitation] Built dataset for 'cube': {X.shape[0]} samples, obs_dim={X.shape[1]}")
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

    # ---------- Helper metrics ----------
    @staticmethod
    def _sign_accuracy(y_pred_norm, y_true_norm):
        # Treat zeros as matches if signs are equal or either is zero
        sp = np.sign(y_pred_norm)
        st = np.sign(y_true_norm)
        return float(np.mean(sp == st))

    def _eps_accuracy(self, y_pred_norm, y_true_norm, eps=None):
        if eps is None:
            eps = self.norm_eps
        return float(np.mean(np.abs(y_pred_norm - y_true_norm) <= eps))

    # ------------- Training with clean-logging/exports -------------
    def train_bc_model(self, task="cube", epochs=150, batch_size=256, lr=1e-3, hidden_sizes=(256, 256), patience=10):
        if not _HAS_TORCH:
            print("[imitation] PyTorch not installed. Please `pip install torch` to train.")
            return False
        X, Y, stats = self._build_bc_dataset(task="cube", frame_stride=3, k_step=3)
        if X is None:
            return False
        x_mean = np.asarray(stats["x_mean"], dtype=np.float32)
        x_std = np.asarray(stats["x_std"], dtype=np.float32)
        self._save_bc_stats("cube", stats)
        N = X.shape[0]
        idx = np.random.permutation(N)
        n_val = max(1, min(max(50, int(0.1 * N)), N - 1))
        val_idx = idx[:n_val]
        train_idx = idx[n_val:]
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
        print(f"[imitation] Training BC for 'cube' | epochs={epochs} | train={X_train_t.shape[0]} | val={X_val_t.shape[0]}")
        best_val = float('inf'); best_ep = 0

        # Logging buffers for optional curves.png
        curve_epochs, curve_train, curve_val, curve_lr = [], [], [], []

        def eval_batchwise_metrics(x_t, y_true_norm_t):
            with torch.no_grad():
                y_pred_norm_t = model(x_t)  # tanh output in [-1,1]
                mse = loss_fn(y_pred_norm_t, y_true_norm_t).item()
                y_pred_np = y_pred_norm_t.cpu().numpy()
                y_true_np = y_true_norm_t.cpu().numpy()
                sign_acc = self._sign_accuracy(y_pred_np, y_true_np)
                eps_acc = self._eps_accuracy(y_pred_np, y_true_np, eps=self.norm_eps)
            return mse, sign_acc, eps_acc

        for ep in range(1, epochs + 1):
            model.train(); perm = torch.randperm(X_train_t.shape[0]); train_loss_sum = 0.0
            # Optional: accumulate accuracy over mini-batches
            train_sign_sum = 0.0; train_eps_sum = 0.0; train_count = 0

            for b in range(steps_per_epoch):
                bi = perm[b*batch_size: (b+1)*batch_size]
                xb = X_train_t[bi]; yb = Y_train_t[bi]
                pred = model(xb); loss = loss_fn(pred, yb)
                opt.zero_grad(); loss.backward(); opt.step()
                train_loss_sum += loss.item() * xb.shape[0]
                # track train accuracies
                with torch.no_grad():
                    y_pred_np = pred.cpu().numpy(); y_true_np = yb.cpu().numpy()
                    train_sign_sum += self._sign_accuracy(y_pred_np, y_true_np) * xb.shape[0]
                    train_eps_sum += self._eps_accuracy(y_pred_np, y_true_np, eps=self.norm_eps) * xb.shape[0]
                    train_count += xb.shape[0]

            train_loss = train_loss_sum / max(1, X_train_t.shape[0])
            train_sign_acc = train_sign_sum / max(1, train_count)
            train_eps_acc = train_eps_sum / max(1, train_count)

            model.eval()
            val_mse, val_sign_acc, val_eps_acc = eval_batchwise_metrics(X_val_t, Y_val_t)
            scheduler.step()

            # Save best model to bc_models for runtime use
            if val_mse < best_val:
                best_val = val_mse; best_ep = ep
                torch.save({"state_dict": model.state_dict(),
                            "input_dim": input_dim,
                            "output_dim": output_dim,
                            "hidden": list(hidden_sizes)}, self._bc_model_path("cube"))

            # Per-epoch logging to logs/training_log.csv with tag cube_clean
            try:
                if append_csv is not None:
                    append_csv(
                        "logs/training_log.csv",
                        ["tag","epoch","train_mse","val_mse","train_sign_acc","val_sign_acc","train_eps_acc","val_eps_acc","lr","time"],
                        {"tag": "cube_clean", "epoch": ep,
                         "train_mse": float(train_loss), "val_mse": float(val_mse),
                         "train_sign_acc": float(train_sign_acc), "val_sign_acc": float(val_sign_acc),
                         "train_eps_acc": float(train_eps_acc), "val_eps_acc": float(val_eps_acc),
                         "lr": float(scheduler.get_last_lr()[0]), "time": datetime.now().isoformat()}
                    )
            except Exception as e:
                print(f"[metrics] Failed to append training_log.csv: {e}")

            # Buffers for curves
            curve_epochs.append(ep); curve_train.append(float(train_loss)); curve_val.append(float(val_mse)); curve_lr.append(float(scheduler.get_last_lr()[0]))

            if ep % 5 == 0 or ep == 1 or ep == epochs:
                print(f"  [ep {ep:03d}] train_mse={train_loss:.6f} | val_mse={val_mse:.6f} | "
                      f"train_sign_acc={train_sign_acc:.3f} | val_sign_acc={val_sign_acc:.3f} | "
                      f"train_eps_acc={train_eps_acc:.3f} | val_eps_acc={val_eps_acc:.3f} | "
                      f"lr={scheduler.get_last_lr()[0]:.2e}")
            if ep - best_ep >= patience:
                print(f"  [early stop] no val improvement for {patience} epochs (best at ep {best_ep})")
                break

        print(f"[imitation] Saved best BC (runtime) to {self._bc_model_path('cube')} (best_val={best_val:.6f})")

        # Standardized CLEAN exports
        try:
            ckpt = torch.load(self._bc_model_path("cube"), map_location="cpu")
            model.load_state_dict(ckpt["state_dict"])
            model.eval()
            clean_dir = "logs/cube_clean"
            if ensure_dir is not None:
                ensure_dir(clean_dir)
            clean_ckpt = os.path.join(clean_dir, "clean_model.pt")

            if save_checkpoint is not None:
                save_checkpoint(model, clean_ckpt, extra={"input_dim": input_dim, "output_dim": output_dim, "hidden": list(hidden_sizes)})
                print(f"[metrics] Clean model checkpoint saved to {clean_ckpt}")

            # Spectral stats -> metrics CSV and compact JSON
            try:
                if spectral_norm_stats is not None and append_csv is not None and write_json is not None and read_json is not None:
                    spec = spectral_norm_stats(model)
                    append_csv(
                        "logs/metrics_summary.csv",
                        ["model","spectral_norm_mean","spectral_norm_max","cka_vs_clean","pruning_auc","cube_success_rate","time"],
                        {"model": "cube_clean",
                         "spectral_norm_mean": spec["spectral_norm_mean"],
                         "spectral_norm_max": spec["spectral_norm_max"],
                         "cka_vs_clean": "", "pruning_auc": "", "cube_success_rate": "",
                         "time": datetime.now().isoformat()}
                    )
                    table_path = "logs/table_compact.json"
                    table = read_json(table_path, default={}) or {}
                    table.setdefault("cube_clean", {}).update(spec)
                    write_json(table_path, table)
                    print("[metrics] Wrote spectral stats for cube_clean.")
            except Exception as e:
                print(f"[metrics] Spectral stats write failed: {e}")

            # Feature export for CKA
            try:
                if collect_features_sequential_penultimate is not None:
                    from torch.utils.data import TensorDataset, DataLoader as TorchLoader
                    val_loader_feats = TorchLoader(
                        TensorDataset(torch.from_numpy(((X_val - x_mean) / x_std)).float(),
                                      torch.from_numpy(Y_val).float()),
                        batch_size=256, shuffle=False
                    )
                    feats_c = collect_features_sequential_penultimate(model, val_loader_feats, device="cpu", max_batches=10)
                    torch.save({"features": feats_c}, os.path.join(clean_dir, "clean_features.pt"))
                    print("[metrics] Clean features exported to logs/cube_clean/clean_features.pt")
            except Exception as e:
                print(f"[metrics] Feature export failed: {e}")

            # Pruning robustness AUC for clean
            try:
                if pruning_robustness_auc is not None:
                    from torch.utils.data import TensorDataset, DataLoader as TorchLoader
                    val_loader_auc = TorchLoader(
                        TensorDataset(torch.from_numpy(((X_val - x_mean) / x_std)).float(),
                                      torch.from_numpy(Y_val).float()),
                        batch_size=256, shuffle=False
                    )
                    def model_ctor():
                        return nn.Sequential(
                            nn.Linear(input_dim, hidden_sizes[0]), nn.ReLU(),
                            nn.Linear(hidden_sizes[0], hidden_sizes[1]), nn.ReLU(),
                            nn.Linear(hidden_sizes[1], output_dim), nn.Tanh()
                        )
                    auc_info = pruning_robustness_auc(model_ctor, torch.load(self._bc_model_path("cube"), map_location="cpu"), val_loader_auc, device="cpu")
                    if append_csv is not None:
                        append_csv(
                            "logs/metrics_summary.csv",
                            ["model","spectral_norm_mean","spectral_norm_max","cka_vs_clean","pruning_auc","cube_success_rate","time"],
                            {"model": "cube_clean", "spectral_norm_mean": "", "spectral_norm_max": "", "cka_vs_clean": "", "pruning_auc": float(auc_info["pruning_auc"]), "cube_success_rate": "", "time": datetime.now().isoformat()}
                        )
                    if write_json is not None and read_json is not None:
                        table_path = "logs/table_compact.json"
                        table = read_json(table_path, default={}) or {}
                        table.setdefault("cube_clean", {})["pruning_auc"] = float(auc_info["pruning_auc"])
                        write_json(table_path, table)
                    print(f"[metrics] Pruning AUC (clean): {auc_info['pruning_auc']:.4f}")
            except Exception as e:
                print(f"[metrics] Pruning AUC export failed: {e}")

            # Optional: curves.png
            try:
                if _HAS_MPL:
                    if ensure_dir is not None:
                        ensure_dir("logs")
                    plt.figure(figsize=(7,4))
                    plt.plot(curve_epochs, curve_train, label="train_mse")
                    plt.plot(curve_epochs, curve_val, label="val_mse")
                    ax2 = plt.gca().twinx()
                    ax2.plot(curve_epochs, curve_lr, color="gray", alpha=0.4, linestyle="--", label="lr")
                    plt.title("Cube Clean Training Curves")
                    plt.xlabel("Epoch"); plt.ylabel("MSE")
                    plt.tight_layout()
                    plt.legend(loc="upper right")
                    plt.savefig("logs/curves.png", dpi=140)
                    plt.close()
                    print("[plot] Saved curves to logs/curves.png")
            except Exception as e:
                print(f"[plot] Failed to plot curves: {e}")

        except Exception as e:
            print(f"[metrics] Clean exports failed: {e}")

        return True

    def load_bc_policy(self, task="cube"):
        if not _HAS_TORCH:
            return False
        model_path = self._bc_model_path("cube")
        stats = self._load_bc_stats("cube")
        if not os.path.isfile(model_path) or stats is None:
            print(f"[policy] No BC model/stats for 'cube'.")
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
        self.bc_models["cube"] = model
        self.bc_stats["cube"] = stats
        print(f"[policy] Loaded BC model for 'cube' from {model_path}")
        return True

    # ---------- Hybrid BC runtime (cube-only) ----------

    def run_bc_policy(self, task="cube", max_steps=600):
        if not _HAS_TORCH:
            print("[policy] PyTorch not installed; cannot run policy.")
            return
        model = self.bc_models.get("cube", None)
        stats = self.bc_stats.get("cube", None)
        if model is None or stats is None:
            if not self.load_bc_policy("cube"):
                return
            model = self.bc_models.get("cube", None)
            stats = self.bc_stats.get("cube", None)
            if model is None or stats is None:
                print(f"[policy] Missing model/stats for 'cube' after load; aborting BC run.")
                return
        if not isinstance(stats, dict) or "x_mean" not in stats or "x_std" not in stats:
            print(f"[policy] Stats missing keys for 'cube'; aborting BC run.")
            return
        x_mean = np.asarray(stats["x_mean"], dtype=np.float32)
        x_std = np.asarray(stats["x_std"], dtype=np.float32)
        action_max = self.action_max_delta.astype(np.float32)

        phase = "approach"  # approach -> descend -> close -> lift
        close_done_steps = 0

        near_xy_thresh = 0.06
        near_z_thresh = 0.035
        latch_duration_s = 0.5
        min_progress = 0.0010
        stagnation_limit = 35
        latch_steps_default = max(2, int(latch_duration_s / max(1e-4, self.policy_iter_dt)))

        prev_dist = None
        stagnation_steps = 0

        for step in range(max_steps):
            # Observe and predict action
            obs = self._obs_from_live("cube").astype(np.float32)
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
            tgt = self._target_pos("cube")

            dxy = float(np.linalg.norm(np.array(ee[:2]) - np.array(tgt[:2])))
            dz = float(abs(ee[2] - tgt[2]))
            dist3d = float(np.linalg.norm(ee - tgt))

            # Phase logic (cube)
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

            # Short-time latch heuristic to ensure closure near contact
            if dxy < near_xy_thresh and dz < near_z_thresh and self.grasp_latch == 0 and phase not in ("close", "lift"):
                self.grasp_latch = latch_steps_default

            if self.grasp_latch > 0:
                try:
                    self.franka.control_dofs_force(np.array([-3.0, -3.0]), self.fingers_dof)
                except Exception:
                    pass
                self.grasp_latch -= 1
            else:
                if phase in ("close", "lift"):
                    try:
                        self.franka.control_dofs_force(np.array([-0.9, -0.9]), self.fingers_dof)
                    except Exception:
                        pass
                else:
                    try:
                        self.franka.control_dofs_position(np.array([0.06, 0.06]), self.fingers_dof)
                    except Exception:
                        pass

            # Reduce motion magnitude near target to avoid nudging and overshoot
            delta_used = base_delta.copy()
            if (dxy < 0.08 and dz < 0.06):
                delta_used *= 0.6

            # Command arm only; gripper handled above
            qcmd = qcur.copy()
            if qcur.shape[0] >= 9:
                qcmd[:9] = qcmd[:9] + delta_used
            else:
                qcmd[:qcur.shape[0]] = qcmd[:qcur.shape[0]] + delta_used[:qcur.shape[0]]
            try:
                self.franka.control_dofs_position(qcmd[:7], self.motors_dof)
            except Exception:
                pass

            # Progress fallback to scripted expert
            if prev_dist is not None:
                progress = prev_dist - dist3d
                if progress < min_progress:
                    stagnation_steps += 1
                else:
                    stagnation_steps = 0
            prev_dist = dist3d
            if stagnation_steps >= stagnation_limit:
                print(f"[policy] No progress; fallback to scripted cube.")
                for frame in self.pick_and_place_cube():
                    yield frame
                return

            for _ in range(self.iter_steps_per_action):
                self.scene.step()
            yield self.capture_frame("cube", "policy_bc")

    # ---------- Success metrics ----------
    def _compute_cube_success_rate(self, episodes, tol_xy=0.02, min_z=0.12):
        """
        Success if the final cube position in an episode is within tol_xy in x,y
        w.r.t. the target pose (0.65, 0.4, ~), and on table (z >= min_z).
        Uses the last frame of each episode.
        """
        if not episodes:
            return 0.0
        target_xy = np.array([0.65, 0.40], dtype=np.float32)
        succ = 0
        for ep in episodes:
            frames = ep.get("frames", [])
            if not frames:
                continue
            last = frames[-1]
            c = np.asarray(last.get("cube_pos", [0, 0, 0]), dtype=np.float32)
            dxy = np.linalg.norm(c[:2] - target_xy)
            if dxy <= tol_xy and c[2] >= float(min_z):
                succ += 1
        return succ / max(1, len(episodes))

    # ---------- Orchestrator ----------

    def main_collect_train_rollout(self, demos=14, train_epochs=150, bc_rollouts=6):
        # 1) 14 scripted demos (cube-only)
        self.collect_fixed_dataset(num_episodes=demos, use_bc_policy=False)
        # 2) Train BC with clean logging/exports
        if _HAS_TORCH:
            self.train_bc_model(task="cube", epochs=train_epochs)
        else:
            print("[imitation] Install PyTorch to train BC models: pip install torch --index-url https://download.pytorch.org/whl/cpu")
        # 3) 6 learned (BC) rollouts
        self.collect_fixed_dataset(num_episodes=bc_rollouts, use_bc_policy=True)


def main():
    collector = FixedFrankaCollector(
        output_dir="franka_dataset_cube",
        use_ai_vision=False,       # not needed for cube-only
        yolo_weights="yolov8n.pt"  # ignored if use_ai_vision=False
    )
    # Produce: 14 scripted demos, train, then 6 learned rollouts (episodes 0..19)
    collector.main_collect_train_rollout(demos=14, train_epochs=150, bc_rollouts=6)


if __name__ == "__main__":
    main()
