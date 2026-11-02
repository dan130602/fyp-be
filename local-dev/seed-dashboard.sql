-- Fake calls for the officer dashboard, for local development only. Never run this against AWS.
-- Requires the Flyway migrations (run them by starting the backend once) and the seeded advisors.
-- Re-running replaces only this script's rows, whose ids start with 00000000-0000-0000-000[2-6].
--
--   docker exec -i ingestion-db psql -U postgres -d ingestion -v ON_ERROR_STOP=1 < local-dev/seed-dashboard.sql
BEGIN;

DELETE FROM t_checklist_decision WHERE id::text LIKE '00000000-0000-0000-0005-%';
DELETE FROM t_langgraph_output WHERE id::text LIKE '00000000-0000-0000-0004-%';
DELETE FROM t_checklist WHERE id::text LIKE '00000000-0000-0000-0003-%';
DELETE FROM t_media_asset WHERE id::text LIKE '00000000-0000-0000-0002-%';
DELETE FROM t_user WHERE id::text LIKE '00000000-0000-0000-0006-%';

INSERT INTO t_user (id, cognito_sub, email, name, role)
VALUES ('00000000-0000-0000-0006-000000000001', 'seed-dashboard-uploader', 'seed.uploader@clearcall-bank.example',
        'Seed Uploader', 'COMPLIANCE_OFFICER');

DO $$
DECLARE
    uploader CONSTANT uuid := '00000000-0000-0000-0006-000000000001';
    questions CONSTANT text[] := ARRAY[
        'Was the caller''s identity verified before account details were discussed?',
        'Was the client''s instruction clear and explicitly confirmed?',
        'Were the key risks and costs of the product explained?',
        'Is the call free of money laundering or suspicious instruction indicators?'];
    reasons CONSTANT jsonb := '{
        "PASS": "The transcript shows this was handled on the call.",
        "FAIL": "The advisor moved on without doing this.",
        "PARTIAL": "This was only partly covered on the call.",
        "CANNOT_DETERMINE": "The transcript does not show either way.",
        "NOT_APPLICABLE": "This did not apply to the call."}';
    segment_text CONSTANT text[] := ARRAY[
        'Good morning, can I confirm your date of birth and account number?',
        'I would like to buy fifty thousand dollars of the income fund.',
        'Just to confirm, that is fifty thousand dollars of the income fund?',
        'Yes, please go ahead with that.'];
    i int;
    idx text;
    media_id uuid;
    checklist_id uuid;
    advisor_slot int;
    advisor uuid;
    day_offset int;
    created timestamp;
    media_status text;
    checklist_status text;
    verdicts text[];
    q int;
    results jsonb;
    checklist jsonb;
    segments jsonb;
    red_flags jsonb;
    has_problem boolean;
    escalated boolean;
    held boolean;
    output jsonb;
BEGIN
    segments := '[]';
    FOR q IN 1..4 LOOP
        segments := segments || jsonb_build_object('id', 'seg_000' || q,
            'speaker', 'spk_' || (q % 2), 'role', CASE WHEN q % 2 = 1 THEN 'Advisor' ELSE 'Client' END,
            'turn', q, 'start', (q - 1) * 15, 'end', q * 15, 'text', segment_text[q]);
    END LOOP;

    checklist := '[]';
    FOR q IN 1..4 LOOP
        checklist := checklist || jsonb_build_object('id', 'Q' || q, 'question', questions[q]);
    END LOOP;

    FOR i IN 1..72 LOOP
        idx := lpad(i::text, 12, '0');
        media_id := ('00000000-0000-0000-0002-' || idx)::uuid;
        checklist_id := ('00000000-0000-0000-0003-' || idx)::uuid;
        day_offset := (i * 5) % 60;
        created := now()::timestamp - make_interval(days => day_offset, hours => i % 9);

        -- Eight advisors with different call patterns; every 18th call predates advisor selection.
        advisor_slot := i % 8;
        advisor := CASE WHEN i % 18 = 0 THEN NULL
                        ELSE ('00000000-0000-0000-0001-' || lpad(((advisor_slot * 2) + 1)::text, 12, '0'))::uuid END;

        media_status := CASE WHEN i % 23 = 0 THEN 'FAILED' WHEN i % 19 = 0 THEN 'PROCESSING' ELSE 'COMPLETED' END;
        checklist_status := CASE WHEN i % 5 = 0 THEN 'SUBMITTED' ELSE 'REVIEWED' END;
        IF media_status <> 'COMPLETED' THEN
            checklist_status := 'SUBMITTED';
        END IF;

        INSERT INTO t_media_asset (id, user_id, advisor_id, object_key, original_filename, content_type, file_size_bytes,
                                   status, failure_reason, created_at, updated_at)
        VALUES (media_id, uploader, advisor, 'seed/call-' || i || '.mp3', 'seed-call-' || lpad(i::text, 3, '0') || '.mp3',
                'audio/mpeg', 1048576, media_status,
                CASE WHEN media_status = 'FAILED' THEN 'Seeded transcription failure' END, created, created);

        INSERT INTO t_checklist (id, media_asset_id, submitted_by, checklist, status, reviewed_by, reviewed_at, created_at, updated_at)
        VALUES (checklist_id, media_id, uploader, checklist, checklist_status,
                CASE WHEN checklist_status = 'REVIEWED' THEN uploader END,
                CASE WHEN checklist_status = 'REVIEWED' THEN created + interval '1 day' END, created, created);

        CONTINUE WHEN media_status <> 'COMPLETED';

        IF i % 29 = 0 THEN
            INSERT INTO t_langgraph_output (id, checklist_id, output, created_at, updated_at)
            VALUES (('00000000-0000-0000-0004-' || idx)::uuid, checklist_id,
                    '{"__error__": "Seeded pipeline failure"}', created, created);
            CONTINUE;
        END IF;

        verdicts := ARRAY['PASS', 'PASS', 'PASS', 'PASS'];
        -- Later advisor slots breach less often, so the advisor table has a spread of rates.
        IF i % (advisor_slot + 2) = 0 THEN verdicts[2] := 'FAIL'; END IF;
        IF i % ((advisor_slot + 2) * 2) = 0 THEN verdicts[3] := 'PARTIAL'; END IF;
        IF i % 7 = 0 THEN verdicts[1] := 'CANNOT_DETERMINE'; END IF;
        IF i % 6 = 0 THEN verdicts[3] := 'NOT_APPLICABLE'; END IF;
        IF i % 25 = 0 THEN verdicts[4] := 'FAIL'; END IF;

        results := '[]';
        FOR q IN 1..4 LOOP
            results := results || jsonb_build_object('item_id', 'Q' || q, 'status', verdicts[q],
                'execution_status', 'success', 'confidence', 0.85, 'reasoning', reasons ->> verdicts[q],
                'evidence', CASE WHEN verdicts[q] IN ('PASS', 'FAIL', 'PARTIAL')
                    THEN jsonb_build_array(jsonb_build_object('segment_id', 'seg_000' || q, 'start_char', 0,
                        'end_char', length(segment_text[q]), 'verified', true))
                    ELSE '[]'::jsonb END);
        END LOOP;

        red_flags := CASE WHEN verdicts[4] = 'FAIL' THEN jsonb_build_array(jsonb_build_object(
            'category', 'suspicious_instruction', 'severity', 'high',
            'description', 'Client asked to split a transfer into smaller amounts.', 'evidence', '[]'::jsonb))
            ELSE '[]'::jsonb END;
        has_problem := 'FAIL' = ANY (verdicts) OR 'PARTIAL' = ANY (verdicts);
        escalated := has_problem;
        held := verdicts[4] = 'FAIL';

        output := jsonb_build_object('call_id', 'seed-call-' || i, 'transcript', 'Seeded transcript',
            'segments', segments, 'checklist', checklist, 'final_results', results, 'final_red_flags', red_flags,
            'actions', '[]'::jsonb,
            'summary', jsonb_build_object('total', 4, 'needs_review', has_problem,
                'escalation', jsonb_build_object('required', escalated, 'hold_transaction', held,
                    'source', CASE WHEN escalated THEN 'rules' ELSE 'none' END, 'rule_ids', '[]'::jsonb, 'reasons', '[]'::jsonb),
                'actions_by_owner', CASE WHEN escalated THEN '{"Compliance": 1, "Advisor": 1}'::jsonb ELSE '{"Operations": 1}'::jsonb END));

        INSERT INTO t_langgraph_output (id, checklist_id, output, created_at, updated_at)
        VALUES (('00000000-0000-0000-0004-' || idx)::uuid, checklist_id, output, created, created);

        -- Some reviewers disagreed with the AI's failure on the second question.
        IF checklist_status = 'REVIEWED' AND verdicts[2] = 'FAIL' AND i % 3 = 0 THEN
            INSERT INTO t_checklist_decision (id, checklist_id, checklist_item_id, ai_verdict, decision, comment, decided_by, decided_at)
            VALUES (('00000000-0000-0000-0005-' || idx)::uuid, checklist_id, 'Q2', 'FAIL', 'PASS',
                    'Confirmed earlier in the recording.', uploader, created + interval '1 day');
        END IF;
    END LOOP;
END $$;

COMMIT;

SELECT status, count(*) FROM t_checklist WHERE id::text LIKE '00000000-0000-0000-0003-%' GROUP BY status ORDER BY status;
