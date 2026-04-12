#!/usr/bin/env python3

import json
from pathlib import Path

import numpy as np
import pinocchio as pin
import qpsolvers
from scipy.spatial.transform import Rotation

import pink
from pink import solve_ik
from pink.tasks import DampingTask, FrameTask, PostureTask


ROOT = Path("/home/tp2/Documents/kejia/blender/scripts/pink_backend")
URDF_PATH = ROOT / "panda_blender_left.urdf"
FRAME_NAME = "tcp_fingertips"


def build_robot():
    return pin.RobotWrapper.BuildFromURDF(
        filename=str(URDF_PATH),
        package_dirs=[str(ROOT)],
        root_joint=None,
    )


def pose_to_matrix(position, quaternion_xyzw):
    matrix = np.eye(4)
    matrix[:3, :3] = Rotation.from_quat(quaternion_xyzw).as_matrix()
    matrix[:3, 3] = np.array(position, dtype=float)
    return matrix


def matrix_to_pose(matrix):
    return {
        "position": matrix[:3, 3].tolist(),
        "quaternion_xyzw": Rotation.from_matrix(matrix[:3, :3]).as_quat().tolist(),
    }


class PinkPandaSolver:
    def __init__(self):
        self.robot = build_robot()
        self.solver = "daqp" if "daqp" in qpsolvers.available_solvers else qpsolvers.available_solvers[0]

    def configuration_from_q(self, q7):
        q = np.zeros(self.robot.model.nq)
        q[:7] = np.array(q7, dtype=float)
        configuration = pink.Configuration(self.robot.model, self.robot.data, q)
        configuration.update(q)
        return configuration

    def solve(
        self,
        q_init,
        current_ee_position,
        current_ee_quaternion_xyzw,
        target_position,
        target_quaternion_xyzw,
        iterations=30,
        dt=0.02,
        posture_cost=1e-3,
        damping_cost=1e-3,
    ):
        configuration = self.configuration_from_q(q_init)
        current_pink = configuration.get_transform_frame_to_world(FRAME_NAME).np
        current_blender = pose_to_matrix(current_ee_position, current_ee_quaternion_xyzw)
        world_from_pink = current_blender @ np.linalg.inv(current_pink)

        target_blender = pose_to_matrix(target_position, target_quaternion_xyzw)
        target_pink = np.linalg.inv(world_from_pink) @ target_blender

        frame_task = FrameTask(
            FRAME_NAME,
            position_cost=1.0,
            orientation_cost=1.0,
            lm_damping=1.0,
        )
        posture_task = PostureTask(cost=posture_cost)
        damping_task = DampingTask(cost=damping_cost)
        tasks = [frame_task, posture_task, damping_task]

        frame_task.set_target_from_configuration(configuration)
        posture_task.set_target_from_configuration(configuration)
        frame_task.transform_target_to_world.translation = target_pink[:3, 3]
        frame_task.transform_target_to_world.rotation = target_pink[:3, :3]

        for _ in range(iterations):
            velocity = solve_ik(configuration, tasks, dt, solver=self.solver)
            configuration.integrate_inplace(velocity, dt)

        solved_pink = configuration.get_transform_frame_to_world(FRAME_NAME).np
        solved_blender = world_from_pink @ solved_pink

        pos_err = np.linalg.norm(solved_blender[:3, 3] - target_blender[:3, 3])
        rot_err = (
            Rotation.from_matrix(target_blender[:3, :3])
            * Rotation.from_matrix(solved_blender[:3, :3]).inv()
        ).magnitude()

        return {
            "q": configuration.q[:7].tolist(),
            "pos_error": float(pos_err),
            "rot_error": float(rot_err),
            "solver": self.solver,
            "solved_ee": matrix_to_pose(solved_blender),
        }

    def calibrated_position_jacobian(
        self,
        q_init,
        current_ee_position,
        current_ee_quaternion_xyzw,
        eps=1e-4,
    ):
        configuration = self.configuration_from_q(q_init)
        current_pink = configuration.get_transform_frame_to_world(FRAME_NAME).np
        current_blender = pose_to_matrix(current_ee_position, current_ee_quaternion_xyzw)
        world_from_pink = current_blender @ np.linalg.inv(current_pink)

        base = world_from_pink @ current_pink
        columns = []
        q_init = list(q_init)
        for joint_index in range(7):
            q = q_init.copy()
            q[joint_index] += eps
            perturbed = self.configuration_from_q(q)
            perturbed_world = world_from_pink @ perturbed.get_transform_frame_to_world(FRAME_NAME).np
            dp = (perturbed_world[:3, 3] - base[:3, 3]) / eps
            columns.append(dp.tolist())
        return {
            "columns": columns,
            "eps": eps,
        }


def solve_once(payload):
    solver = PinkPandaSolver()
    return solver.solve(
        q_init=payload["q"],
        current_ee_position=payload["current_ee"]["position"],
        current_ee_quaternion_xyzw=payload["current_ee"]["quaternion_xyzw"],
        target_position=payload["target"]["position"],
        target_quaternion_xyzw=payload["target"]["quaternion_xyzw"],
        iterations=payload.get("iterations", 30),
        dt=payload.get("dt", 0.02),
        posture_cost=payload.get("posture_cost", 1e-3),
        damping_cost=payload.get("damping_cost", 1e-3),
    )


def jacobian_once(payload):
    solver = PinkPandaSolver()
    return solver.calibrated_position_jacobian(
        q_init=payload["q"],
        current_ee_position=payload["current_ee"]["position"],
        current_ee_quaternion_xyzw=payload["current_ee"]["quaternion_xyzw"],
        eps=payload.get("eps", 1e-4),
    )


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input-json", required=True)
    args = parser.parse_args()
    payload = json.loads(Path(args.input_json).read_text())
    print(json.dumps(solve_once(payload)))


if __name__ == "__main__":
    main()
