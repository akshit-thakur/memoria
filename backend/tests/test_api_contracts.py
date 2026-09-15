"""Regression checks for frontend-visible FastAPI contracts."""

from __future__ import annotations

import asyncio
import inspect
from pathlib import Path

from routers import (
    ALL_ROUTERS,
    _hydrate_scene_neighbors,
    _model_profile,
    faces_clear_all,
    faces_get_cluster,
    faces_list_clusters,
    faces_similar,
    library_scan_folder,
    search_similar,
    search_text,
)


def _registered_routes() -> set[tuple[str, str]]:
    return {
        (route.path, method)
        for router in ALL_ROUTERS
        for route in router.routes
        for method in getattr(route, "methods", set())
    }


def test_processing_model_management_routes_are_registered():
    routes = _registered_routes()
    expected = {
        ("/api/v1/processing/config", "GET"),
        ("/api/v1/processing/models", "GET"),
        ("/api/v1/processing/reindex/scene", "POST"),
        ("/api/v1/processing/reindex/{job_id}", "GET"),
        ("/api/v1/processing/reindex/{job_id}/cancel", "POST"),
        ("/api/v1/processing/benchmarks", "GET"),
    }
    assert expected <= routes


def test_persisted_model_rows_keep_frontend_compatibility_aliases():
    profile = _model_profile({
        "model_key": "scene/test/model",
        "kind": "scene",
        "driver": "test",
        "model_id": "test/model",
        "embedding_dimension": 3,
        "input_resolution": [32, 32],
        "default_precision": "fp32",
        "provider_preferences": ["cpu"],
    })
    assert profile is not None
    assert profile["model_name"] == "test/model"
    assert profile["embedding_dim"] == 3
    assert profile["preferred_providers"] == ["cpu"]
    assert profile["precision"] == "fp32"


def test_search_handlers_use_active_namespaced_scene_helpers():
    text_source = inspect.getsource(search_text)
    similar_source = inspect.getsource(search_similar)
    assert "get_active_model_async" in text_source
    assert "get_active_scene_embedding_async" in similar_source
    assert "query_scene_neighbors_async" in text_source
    assert "query_scene_neighbors_async" in similar_source
    assert "scene_embedding <=>" not in text_source
    assert "scene_embedding <=>" not in similar_source


def test_face_reads_are_scoped_to_the_active_model_namespace():
    list_source = inspect.getsource(faces_list_clusters)
    detail_source = inspect.getsource(faces_get_cluster)
    similar_source = inspect.getsource(faces_similar)
    assert "state.active_model_key = fe.embedding_model_key" in list_source
    assert "state.active_model_key = fe.embedding_model_key" in detail_source
    assert "state.active_model_key = fe.embedding_model_key" in similar_source
    assert "model_predicate" in similar_source


def test_face_clear_removes_completion_markers_so_index_can_rebuild():
    clear_source = inspect.getsource(faces_clear_all)
    assert "DELETE FROM ml_processing_state" in clear_source
    assert "kind = 'face-embedding'" in clear_source


def test_folder_picker_scan_contract_uses_a_bounded_relative_path_sample():
    route_source = inspect.getsource(library_scan_folder)
    assert "MAX_LIBRARY_SCAN_BODY_BYTES" in route_source
    assert "validate_picked_scan_request" in route_source
    assert '"matched_files": total_files' in route_source
    assert '"sampled_files": len(files)' in route_source

    component_path = (
        Path(__file__).resolve().parents[2]
        / "frontend/src/components/LibraryFolderPicker.tsx"
    )
    component_source = component_path.read_text(encoding="utf-8")
    assert "JSON.stringify(buildLibraryScanPayload(selection))" in component_source
    assert "total_files: selection.total_files" in component_source
    assert "({ relative_path: entry.relative_path })" in component_source

    picker_path = (
        Path(__file__).resolve().parents[2]
        / "frontend/src/lib/pickDirectory.ts"
    )
    picker_source = picker_path.read_text(encoding="utf-8")
    assert "DIRECTORY_FILE_SAMPLE_LIMIT = 50" in picker_source
    assert "accumulator.sample.length < DIRECTORY_FILE_SAMPLE_LIMIT" in picker_source
    assert "total_files: accumulator.totalFiles" in picker_source
    assert "walkDirectoryHandle(" in picker_source
    assert "accumulator: DirectorySampleAccumulator" in picker_source
    assert picker_source.count("addSampledPath(accumulator") >= 2
    assert "Promise<PickedFileEntry[]>" not in picker_source
    assert "const files: PickedFileEntry[] = []" not in picker_source
    assert "getFile(" not in picker_source


def test_scene_neighbor_hydration_preserves_knn_order():
    class FakeConnection:
        async def fetch(self, _query, media_ids):
            assert media_ids == [9, 4]
            # Deliberately return the reverse of the kNN order.
            return [
                {
                    "id": 4,
                    "filename": "four.jpg",
                    "directory": "/photos",
                    "phash": None,
                    "blur_score": None,
                    "exif_data": {},
                    "indexed_at": None,
                },
                {
                    "id": 9,
                    "filename": "nine.jpg",
                    "directory": "/photos",
                    "phash": None,
                    "blur_score": None,
                    "exif_data": {},
                    "indexed_at": None,
                },
            ]

    items = asyncio.run(_hydrate_scene_neighbors(
        FakeConnection(),
        [
            {"media_id": 9, "distance": 0.1},
            {"media_id": 4, "distance": 0.2},
        ],
    ))
    assert [item["id"] for item in items] == [9, 4]
    assert [item["distance"] for item in items] == [0.1, 0.2]
