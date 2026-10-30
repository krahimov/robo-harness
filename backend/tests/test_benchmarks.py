import threading

import pytest

from studio.adapters import astra
from studio.adapters.franka import available
from studio.benchmarks import BENCHMARKS
from studio.schemas import Scene
from studio.task_api import TaskCreate, bind_task, get_task, list_tasks, measure_task

M = astra.Motion


def test_benchmark_presets_are_human_owned_franka_tasks_with_disclosed_deviations():
    listed = {task["id"]: task for task in list_tasks()}
    for id in BENCHMARKS:
        task = listed[id]
        TaskCreate(**{key: task[key] for key in ("name", "description", "requirements", "operations")})
        assert task["owner"] == "human" and task["required_embodiment"] == "franka-panda-v1"
        assert task["benchmark"]["suite"] == "robomimic" and task["benchmark"]["deviations"]
        assert all(r["required"] for r in task["requirements"])
        bound = bind_task(Scene(**task["suggested_scene"]), task)
        assert bound["embodiment"] == "franka-panda-v1"
        with pytest.raises(ValueError, match="Franka Panda"):
            bind_task(Scene(), task)


PROGRAMS = {
    "robomimic-lift": [M(target="cube", offset=[0, 0, .12], yaw=0, closed=False, seconds=2),
        M(target="cube", offset=[0, 0, 0], yaw=0, closed=False, seconds=2),
        M(target="cube", offset=[0, 0, 0], yaw=0, closed=True, seconds=1),
        M(target="gripper", offset=[0, 0, .12], yaw=0, closed=True, seconds=2),
        M(target="gripper", offset=[0, 0, 0], yaw=0, closed=True, seconds=1)],
    "robomimic-can": [M(target="can", offset=[0, 0, .14], yaw=0, closed=False, seconds=2),
        M(target="can", offset=[0, 0, .02], yaw=0, closed=False, seconds=2),
        M(target="can", offset=[0, 0, .02], yaw=0, closed=True, seconds=1),
        M(target="gripper", offset=[0, 0, .16], yaw=0, closed=True, seconds=2),
        M(target="goal_can_in_bin", offset=[0, 0, .18], yaw=0, closed=True, seconds=2.5),
        M(target="goal_can_in_bin", offset=[0, 0, .02], yaw=0, closed=True, seconds=1.5),
        M(target="gripper", offset=[0, 0, 0], yaw=0, closed=False, seconds=1),
        M(target="gripper", offset=[0, 0, .12], yaw=0, closed=False, seconds=1)],
}
IDLE = [M(target="gripper", offset=[0, 0, 0], yaw=0, closed=False, seconds=.5)] * 4


@pytest.mark.skipif(not available(), reason="Run integrations/icrt/prepare_franka.py to install the licensed assets")
@pytest.mark.parametrize("id", list(BENCHMARKS))
def test_benchmark_is_achievable_and_not_trivially_passed(id):
    task = get_task(id)
    scene = Scene(**{**task["suggested_scene"], "seed": 7})
    bound = bind_task(scene, task)
    solved = astra.execute(scene, bound, astra.Program(motions=PROGRAMS[id]), threading.Event())
    assert solved["execution_error"] is None
    assert measure_task(solved, {"require_review": False})["status"] == "pass"
    idle = astra.execute(scene, bound, astra.Program(motions=IDLE), threading.Event())
    assert measure_task(idle, {"require_review": False})["status"] == "fail"
