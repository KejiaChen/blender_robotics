#!/usr/bin/env python3

import json
import sys

from pink_ik_solver_right import PinkPandaRightSolver


def main():
    solver = PinkPandaRightSolver()
    print(json.dumps({"status": "ready", "solver": solver.solver}), flush=True)
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        request = json.loads(line)
        command = request.get("command")
        if command == "ping":
            print(json.dumps({"status": "ok"}), flush=True)
            continue
        if command == "solve":
            try:
                result = solver.solve(
                    q_init=request["q"],
                    current_ee_position=request["current_ee"]["position"],
                    current_ee_quaternion_xyzw=request["current_ee"]["quaternion_xyzw"],
                    target_position=request["target"]["position"],
                    target_quaternion_xyzw=request["target"]["quaternion_xyzw"],
                    iterations=request.get("iterations", 30),
                    dt=request.get("dt", 0.02),
                    posture_cost=request.get("posture_cost", 1e-3),
                    damping_cost=request.get("damping_cost", 1e-3),
                )
                print(json.dumps({"status": "ok", "result": result}), flush=True)
            except Exception as exc:
                print(json.dumps({"status": "error", "error": repr(exc)}), flush=True)
            continue
        if command == "quit":
            print(json.dumps({"status": "bye"}), flush=True)
            break
        print(json.dumps({"status": "error", "error": f"unknown command: {command}"}), flush=True)


if __name__ == "__main__":
    main()
