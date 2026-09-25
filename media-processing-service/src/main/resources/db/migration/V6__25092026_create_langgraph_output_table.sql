CREATE TABLE t_langgraph_output(
    id            UUID PRIMARY KEY,
    checklist_id  UUID      NOT NULL UNIQUE REFERENCES t_checklist(id),
    output        JSONB     NOT NULL,
    created_at    TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at    TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);
