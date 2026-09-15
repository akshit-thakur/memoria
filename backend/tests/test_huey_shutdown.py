"""Database-free contracts for durable Huey consumer shutdown."""

from __future__ import annotations

import ast
import configparser
import copy
import types
import unittest
from pathlib import Path
from unittest import mock


_ROOT = Path(__file__).resolve().parents[2]
_WORKERS_PATH = _ROOT / "backend" / "workers" / "__init__.py"
_SUPERVISOR_PATH = _ROOT / "docker" / "supervisord.conf"


def _function_node(name: str) -> ast.FunctionDef:
    tree = ast.parse(_WORKERS_PATH.read_text(), filename=str(_WORKERS_PATH))
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"could not find {name} in {_WORKERS_PATH}")


def _load_function(name: str, namespace: dict[str, object]):
    """Compile one production function without importing the worker runtime."""

    node = copy.deepcopy(_function_node(name))
    node.decorator_list = []
    module = ast.Module(body=[node], type_ignores=[])
    ast.fix_missing_locations(module)
    exec(compile(module, str(_WORKERS_PATH), "exec"), namespace)
    return namespace[name]


class _FakeHuey:
    def __init__(self) -> None:
        self.enqueued: list[object] = []

    def enqueue(self, task: object) -> None:
        self.enqueued.append(task)


class _FakeMetrics:
    def __init__(self) -> None:
        self.counts: list[tuple[str, dict[str, object]]] = []

    def count(self, name: str, **tags: object) -> None:
        self.counts.append((name, tags))


class HueyShutdownContractTests(unittest.TestCase):
    def test_interrupted_task_is_requeued_exactly_once(self) -> None:
        fake_huey = _FakeHuey()
        fake_metrics = _FakeMetrics()
        cleared: list[bool] = []
        task = types.SimpleNamespace(id="task-1", name="process_media")
        task_starts = {task.id: 1.0}
        handler = _load_function(
            "_on_interrupted",
            {
                "huey": fake_huey,
                "metrics": fake_metrics,
                "_signal_logger": mock.Mock(),
                "_task_starts": task_starts,
                "_clear_active_media_marker": lambda: cleared.append(True),
            },
        )

        handler("interrupted", task)

        self.assertEqual(fake_huey.enqueued, [task])
        self.assertNotIn(task.id, task_starts)
        self.assertEqual(cleared, [True])
        self.assertEqual(
            fake_metrics.counts,
            [
                ("worker.task.interrupted", {"task": task.name}),
                ("worker.task.requeued", {"task": task.name}),
            ],
        )

    def test_only_interrupted_handler_reenqueues(self) -> None:
        interrupted = _function_node("_on_interrupted")
        complete = _function_node("_on_complete")

        decorator_names = {
            argument.id
            for decorator in interrupted.decorator_list
            if isinstance(decorator, ast.Call)
            for argument in decorator.args
            if isinstance(argument, ast.Name)
        }
        self.assertEqual(decorator_names, {"SIGNAL_INTERRUPTED"})

        def enqueue_calls(node: ast.AST) -> list[ast.Call]:
            return [
                child
                for child in ast.walk(node)
                if isinstance(child, ast.Call)
                and isinstance(child.func, ast.Attribute)
                and isinstance(child.func.value, ast.Name)
                and child.func.value.id == "huey"
                and child.func.attr == "enqueue"
            ]

        self.assertEqual(len(enqueue_calls(interrupted)), 1)
        self.assertEqual(enqueue_calls(complete), [])

    def test_worker_surrounds_watcher_in_supervisor_lifecycle(self) -> None:
        config = configparser.ConfigParser(interpolation=None)
        config.read(_SUPERVISOR_PATH)

        worker_priority = config.getint("program:worker", "priority")
        watcher_priority = config.getint("program:watcher", "priority")

        # Supervisor starts lower priorities first and stops them last.  This
        # lets the worker drain anything the watcher queues during shutdown.
        self.assertLess(worker_priority, watcher_priority)
        self.assertEqual(config.get("program:worker", "stopsignal"), "TERM")


if __name__ == "__main__":
    unittest.main()
