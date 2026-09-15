from __future__ import annotations

import unittest
from unittest import mock

import database_config


class DatabaseConfigTests(unittest.TestCase):
    def test_explicit_url_wins(self):
        with mock.patch.dict(
            database_config.os.environ,
            {"DATABASE_URL": "postgresql://explicit/db", "DB_HOST": "ignored"},
            clear=True,
        ):
            self.assertEqual(
                database_config.resolve_database_url(),
                "postgresql://explicit/db",
            )

    def test_legacy_postgres_scheme_is_normalized_for_alembic(self):
        with mock.patch.dict(
            database_config.os.environ,
            {"DATABASE_URL": "postgres://legacy:secret@db.example/memoria"},
            clear=True,
        ):
            self.assertEqual(
                database_config.resolve_database_url(),
                "postgresql://legacy:secret@db.example/memoria",
            )

    def test_component_url_encodes_credentials_and_ipv6(self):
        env = {
            "DB_USER": "photo user",
            "DB_PASSWORD": "p@ss/word",
            "DB_HOST": "::1",
            "DB_PORT": "5433",
            "DB_NAME": "photo library",
        }
        with mock.patch.dict(database_config.os.environ, env, clear=True):
            self.assertEqual(
                database_config.resolve_database_url(),
                "postgresql://photo%20user:p%40ss%2Fword@[::1]:5433/photo%20library",
            )


if __name__ == "__main__":
    unittest.main()
