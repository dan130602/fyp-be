-- employee_id is the immutable HR personnel number shown to users; email is display-only
-- because it changes with name changes. department is plain text for now (no lookup table).
CREATE TABLE t_advisor(
    id           UUID PRIMARY KEY,
    employee_id  VARCHAR(7)   NOT NULL UNIQUE,
    full_name    VARCHAR(255) NOT NULL,
    email        VARCHAR(255) NOT NULL UNIQUE,
    department   VARCHAR(100) NOT NULL,
    status       VARCHAR(20)  NOT NULL DEFAULT 'ACTIVE',
    created_at   TIMESTAMP    NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at   TIMESTAMP    NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT chk_advisor_employee_id CHECK (employee_id ~ '^E[0-9]{6}$'),
    CONSTRAINT chk_advisor_status CHECK (status IN ('ACTIVE', 'INACTIVE'))
);

CREATE INDEX idx_advisor_department ON t_advisor(department);
