"""Database-free contract tests for the container schema bootstrap."""

from __future__ import annotations

import ast
import importlib.util
import unittest
from pathlib import Path


_ROOT = Path(__file__).resolve().parents[2]
_BOOTSTRAP_PATH = _ROOT / "docker" / "bootstrap.py"
_MIGRATION_PATH = (
    _ROOT
    / "backend"
    / "alembic"
    / "versions"
    / "012_vector_schema_repair.py"
)


def _load_bootstrap_module():
    spec = importlib.util.spec_from_file_location(
        "memoria_docker_bootstrap_test",
        _BOOTSTRAP_PATH,
    )
    if spec is None or spec.loader is None:
        raise AssertionError("could not load docker/bootstrap.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _FakeResult:
    def __init__(self, problems: list[str]) -> None:
        self._problems = problems

    def fetchone(self):
        return (self._problems,)


class _FakeConnection:
    def __init__(self, problems: list[str]) -> None:
        self.problems = problems
        self.query = ""
        self.params: tuple[str, ...] = ()

    def execute(self, query: str, params: tuple[str, ...]):
        self.query = query
        self.params = params
        return _FakeResult(self.problems)


class DockerBootstrapContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.bootstrap = _load_bootstrap_module()

    def test_repair_revision_is_the_validated_alembic_head(self) -> None:
        tree = ast.parse(_MIGRATION_PATH.read_text(), filename=str(_MIGRATION_PATH))
        assignments: dict[str, object] = {}
        for node in tree.body:
            if (
                isinstance(node, ast.Assign)
                and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id in {"revision", "down_revision"}
            ):
                assignments[node.targets[0].id] = ast.literal_eval(node.value)

        self.assertEqual(assignments["revision"], self.bootstrap.ALEMBIC_HEAD)
        self.assertEqual(assignments["down_revision"], "011_pluggable_ml_models")

    def test_postcondition_covers_every_repaired_vector_object(self) -> None:
        query = self.bootstrap.SCHEMA_POSTCONDITION_SQL
        for marker in (
            "vector(512)",
            "vector(768)",
            "public.scene_embeddings",
            "public.media_scene_hnsw",
            "public.media_dedup_hnsw",
            "public.scene_openclip_vitl14_hnsw",
            "public.scene_openclip_siglip1152_hnsw",
            "public.scene_hf_siglip2_1152_hnsw",
            "public.face_arcface_512_hnsw",
            "public.face_facenet_512_hnsw",
            "public.face_legacy_512_hnsw",
            "face_embedding_vector_dimension_ck",
            "fk_scene_embedding_model_dimension",
        ):
            self.assertIn(marker, query)

    def test_problem_rows_are_returned_without_database_dependencies(self) -> None:
        connection = _FakeConnection(["relation:public.scene_embeddings"])

        problems = self.bootstrap.schema_postcondition_problems(connection)

        self.assertEqual(problems, ["relation:public.scene_embeddings"])
        self.assertEqual(
            connection.params,
            (self.bootstrap.ALEMBIC_HEAD, self.bootstrap.ALEMBIC_HEAD),
        )


if __name__ == "__main__":
    unittest.main()
