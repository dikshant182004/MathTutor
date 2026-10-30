CREATE TABLE IF NOT EXISTS students (
    student_id CHAR(16) PRIMARY KEY,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS threads (
    thread_id TEXT NOT NULL,
    student_id CHAR(16) NOT NULL REFERENCES students(student_id) ON DELETE CASCADE,
    title TEXT NOT NULL DEFAULT '',
    topic TEXT NOT NULL DEFAULT '',
    outcome TEXT NOT NULL DEFAULT 'in_progress',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_active TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (thread_id, student_id)
);

CREATE INDEX IF NOT EXISTS idx_threads_student_activity
    ON threads (student_id, last_active DESC);

CREATE TABLE IF NOT EXISTS conversation_messages (
    message_id UUID PRIMARY KEY,
    thread_id TEXT NOT NULL,
    student_id CHAR(16) NOT NULL,
    role TEXT NOT NULL CHECK (role IN ('user', 'assistant', 'system')),
    content TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    FOREIGN KEY (thread_id, student_id)
        REFERENCES threads(thread_id, student_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_messages_student_thread_time
    ON conversation_messages (student_id, thread_id, created_at, message_id);

CREATE TABLE IF NOT EXISTS tutoring_runs (
    run_id UUID PRIMARY KEY,
    thread_id TEXT NOT NULL,
    student_id CHAR(16) NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('running', 'completed', 'interrupted', 'failed')),
    route TEXT,
    complexity TEXT,
    llm_calls INTEGER NOT NULL DEFAULT 0 CHECK (llm_calls >= 0),
    tool_calls INTEGER NOT NULL DEFAULT 0 CHECK (tool_calls >= 0),
    duration_ms INTEGER CHECK (duration_ms IS NULL OR duration_ms >= 0),
    error_code TEXT,
    started_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    completed_at TIMESTAMPTZ,
    FOREIGN KEY (thread_id, student_id)
        REFERENCES threads(thread_id, student_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_runs_student_started
    ON tutoring_runs (student_id, started_at DESC);

CREATE TABLE IF NOT EXISTS schema_migrations (
    version TEXT PRIMARY KEY,
    applied_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

INSERT INTO schema_migrations(version)
VALUES ('001_conversation_history_analytics')
ON CONFLICT (version) DO NOTHING;
