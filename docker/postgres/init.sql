-- The package and extension use different names: pgvector provides `vector`.
-- This runs only for a newly initialized volume; the application bootstrap
-- repeats both statements on every boot for existing volumes.
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

