CREATE TABLE t_user(
    id          UUID PRIMARY KEY,
    cognito_sub VARCHAR(100) NOT NULL UNIQUE,
    email       VARCHAR(255),
    name        VARCHAR(255),
    role        VARCHAR(30)  NOT NULL,
    status      VARCHAR(20)  NOT NULL DEFAULT 'ACTIVE',
    created_at  TIMESTAMP    NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at  TIMESTAMP    NOT NULL DEFAULT CURRENT_TIMESTAMP
);
