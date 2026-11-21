import copy
import json

import pytest

from meralion_worker import config as cfg

FILE = json.loads(cfg.DEFAULT_CONFIG.read_text(encoding="utf-8"))


def load(path=None, **kwargs):
    kwargs.setdefault("env", {})
    return cfg.load_runtime_config(path, **kwargs)


def variant(**changes):
    data = copy.deepcopy(FILE)
    for dotted, value in changes.items():
        target = data
        *parents, last = dotted.split("__")
        for key in parents:
            target = target[key]
        target[last] = value
    return data


def test_the_frozen_config_loads_with_expected_identity():
    effective = load()
    assert effective.data == FILE
    assert effective.backend == "transformers" and effective.data["decoder"]["settings_id"] == "stock-transformers-v1"
    assert effective.formatter == "percentages-v2+amount-v1" and effective.data["config_id"] == "fg175-m2ax-seq-ap-lmonly-m5-fee2-amt1-nobudget"
    assert effective.data["strategy"] == cfg.STRATEGY_V2 and effective.data["strategy"]["run_cap"] is None
    assert effective.data["strategy"]["run_cap_scope"] == "none"
    assert effective.region == "ap-southeast-1" and effective.output_bucket is None


def test_environment_overrides_and_cli_wins_over_environment():
    env = {"MERALION_BACKEND": "vllm", "MERALION_MODEL_DIR": "/env/model", "TRANSCRIBE_OUTPUT_BUCKET": "bucket-a", "AWS_DEFAULT_REGION": "us-east-1"}
    effective = load(env=env)
    assert effective.backend == "vllm" and str(effective.model_dir).replace("\\", "/") == "/env/model"
    assert effective.output_bucket == "bucket-a" and effective.region == "us-east-1"
    assert load(env={**env, "AWS_REGION": "eu-west-1"}).region == "eu-west-1"           # AWS_REGION beats AWS_DEFAULT_REGION
    effective = load(env=env, cli={"backend": "transformers", "model_dir": "/cli/model"})
    assert effective.backend == "transformers" and str(effective.model_dir).replace("\\", "/") == "/cli/model"


def test_a_backend_override_sets_the_matching_settings_id():
    effective = load(env={"MERALION_BACKEND": "vllm"})
    assert effective.data["decoder"] == {"backend": "vllm", "settings_id": "stock-vllm-package-v1"}
    back = load(env={"MERALION_BACKEND": "vllm"}, cli={"backend": "transformers"})
    assert back.data["decoder"] == {"backend": "transformers", "settings_id": "stock-transformers-v1"}


def test_settings_id_cannot_be_set_on_its_own():
    with pytest.raises(cfg.ConfigError, match="contradicts"):
        load(variant(decoder__settings_id="stock-vllm-package-v1"))
    with pytest.raises(cfg.ConfigError, match="contradicts"):
        load(variant(decoder__settings_id="made-up"))


def test_a_contradictory_file_is_rejected_not_repaired_by_an_override():
    broken = variant(decoder__settings_id="stock-vllm-package-v1")
    with pytest.raises(cfg.ConfigError, match="contradicts"):
        load(broken, env={"MERALION_BACKEND": "transformers"})


def test_unknown_backend_formatter_schema_and_manifest_are_rejected():
    with pytest.raises(cfg.ConfigError, match="unknown backend"):
        load(env={"MERALION_BACKEND": "onnx"})
    with pytest.raises(cfg.ConfigError, match="unknown backend"):
        load(variant(decoder__backend="onnx"))
    with pytest.raises(cfg.ConfigError, match="unknown formatter"):
        load(variant(formatter="percentages-v3"))
    with pytest.raises(cfg.ConfigError, match="output_schema"):
        load(variant(output_schema="v1"))
    with pytest.raises(cfg.ConfigError, match="manifest not found"):
        load(variant(model_manifest="no-such-model"))


@pytest.mark.parametrize("change", [
    {"strategy__window_seconds": 20},
    {"strategy__stride_seconds": 10},
    {"strategy__merge_policy": "m2ax"},
    {"strategy__run_cap": 0.5},
    {"strategy__unresolved_call_cap": 0.2},
    {"strategy__run_cap_scope": "recording"},
    {"strategy__retry_kinds": ["unresolved"]},
    {"strategy__id": "fg-other"},
    {"strategy__reconcile": {"conflict_tiebreak": True, "low_margin_tiebreak": True, "exclude_critical_from_low_margin": True}},
])
def test_any_change_to_the_frozen_strategy_is_rejected(change):
    with pytest.raises(cfg.ConfigError, match="frozen strategy"):
        load(variant(**change))


def test_missing_and_unexpected_keys_are_rejected():
    data = variant()
    del data["formatter"]
    with pytest.raises(cfg.ConfigError, match="missing"):
        load(data)
    data = variant()
    data["extra"] = 1
    with pytest.raises(cfg.ConfigError, match="unexpected"):
        load(data)


def test_formatter_none_is_valid():
    assert load(variant(formatter="none")).formatter == "none"


def test_the_fingerprint_is_stable_and_changes_with_every_field():
    first = load().fingerprint
    assert first == load().fingerprint and len(first) == 64
    assert load(json.loads(json.dumps(FILE))).fingerprint == first                    # same content from a dict
    assert load(env={"MERALION_BACKEND": "vllm"}).fingerprint != first
    assert load(variant(formatter="none")).fingerprint != first
    assert load(variant(config_id="another")).fingerprint != first


def test_environment_facts_are_not_part_of_the_fingerprint():
    first = load().fingerprint
    assert load(env={"MERALION_MODEL_DIR": "/elsewhere", "TRANSCRIBE_OUTPUT_BUCKET": "b", "AWS_REGION": "us-west-2"}).fingerprint == first


def test_the_recorded_form_carries_config_and_fingerprint():
    effective = load()
    record = effective.as_record()
    assert record["config"] == FILE and record["config_fingerprint"] == effective.fingerprint
    record["config"]["formatter"] = "none"                                              # a copy: the effective config is unaffected
    assert effective.data["formatter"] == "percentages-v2+amount-v1"


def test_the_config_file_is_chosen_cli_then_environment_then_default(tmp_path):
    alt = tmp_path / "alt.json"
    alt.write_text(json.dumps(variant(config_id="from-file")), encoding="utf-8")
    other = tmp_path / "other.json"
    other.write_text(json.dumps(variant(config_id="from-cli")), encoding="utf-8")
    assert load(env={"MERALION_CONFIG": str(alt)}).data["config_id"] == "from-file"
    assert load(env={"MERALION_CONFIG": str(alt)}, cli={"config": str(other)}).data["config_id"] == "from-cli"
    with pytest.raises(cfg.ConfigError, match="not found"):
        load(tmp_path / "nope.json")


# ---- the amt1 promotion (Stage 2A): the new default, the preserved legacy config, and what separates them ---------------------------------------------------------------------

LEGACY = json.loads(cfg.LEGACY_CONFIG.read_text(encoding="utf-8"))


BUDGETED = json.loads(cfg.BUDGETED_CONFIG.read_text(encoding="utf-8"))


def test_the_default_is_the_nobudget_config_and_both_earlier_files_are_preserved():
    assert cfg.DEFAULT_CONFIG.name == "fg175-m2ax-seq-ap-lmonly-m5-fee2-amt1-nobudget.json" and cfg.BUDGETED_CONFIG.name == "fg175-m2ax-seq-ap-lmonly-m5-fee2-amt1.json"
    assert cfg.LEGACY_CONFIG.name == "fg175-m2ax-seq-ap-lmonly-m5-fee2.json" and all(p.is_file() for p in (cfg.DEFAULT_CONFIG, cfg.BUDGETED_CONFIG, cfg.LEGACY_CONFIG))
    assert FILE["config_id"] == "fg175-m2ax-seq-ap-lmonly-m5-fee2-amt1-nobudget" and BUDGETED["config_id"] == "fg175-m2ax-seq-ap-lmonly-m5-fee2-amt1" and LEGACY["config_id"] == "fg175-m2ax-seq-ap-lmonly-m5-fee2"


def test_the_budgeted_and_legacy_configs_differ_only_in_config_id_and_formatter():
    assert [k for k in BUDGETED if BUDGETED[k] != LEGACY[k]] == ["config_id", "formatter"] and set(BUDGETED) == set(LEGACY)
    assert LEGACY["formatter"] == "percentages-v2" and BUDGETED["formatter"] == "percentages-v2+amount-v1"
    for unchanged in ("strategy", "decoder", "output_schema", "model_manifest"):
        assert BUDGETED[unchanged] == LEGACY[unchanged]                                      # the strategy, decoding settings, weights and budgets are not touched by the promotion


def test_the_default_differs_from_the_budgeted_config_only_in_config_id_and_the_run_budget_of_the_strategy():
    assert [k for k in FILE if FILE[k] != BUDGETED[k]] == ["config_id", "strategy"] and set(FILE) == set(BUDGETED)
    changed = {k for k in FILE["strategy"] if FILE["strategy"][k] != BUDGETED["strategy"][k]}
    assert changed == {"id", "run_cap", "run_cap_scope"}                                     # the unresolved cap, retry kinds, windows, merge policy and reconcile rules are identical
    assert FILE["strategy"]["run_cap"] is None and FILE["strategy"]["run_cap_scope"] == "none" and FILE["strategy"]["unresolved_call_cap"] == 0.10
    assert FILE["strategy"]["reconcile"] == BUDGETED["strategy"]["reconcile"] and FILE["strategy"]["retry_kinds"] == BUDGETED["strategy"]["retry_kinds"]
    for unchanged in ("decoder", "formatter", "output_schema", "model_manifest"):
        assert FILE[unchanged] == BUDGETED[unchanged]


def test_only_the_two_known_strategies_load_and_any_change_to_either_is_rejected():
    assert set(cfg.STRATEGIES) == {"fg-m2ax-seq-ap-lmonly-v1", "fg-m2ax-seq-ap-lmonly-v2"}
    assert load(cfg.BUDGETED_CONFIG).data["strategy"] == cfg.FROZEN_STRATEGY and load().data["strategy"] == cfg.STRATEGY_V2
    for change in ({"strategy__run_cap": 0.15}, {"strategy__run_cap_scope": "invocation"}, {"strategy__unresolved_call_cap": 0.2}, {"strategy__id": "fg-m2ax-seq-ap-lmonly-v3"}):
        with pytest.raises(cfg.ConfigError, match="frozen strategy"):
            load(variant(**change))
    broken = json.loads(json.dumps(BUDGETED))
    broken["strategy"]["run_cap"] = None                                                      # v1 without its budget is not v1 and not v2
    with pytest.raises(cfg.ConfigError, match="frozen strategy"):
        load(broken)


def test_the_two_strategies_have_different_fingerprints_and_the_new_default_is_distinct():
    assert load().fingerprint != load(cfg.BUDGETED_CONFIG).fingerprint != load(cfg.LEGACY_CONFIG).fingerprint


def test_the_legacy_config_still_loads_and_has_its_own_fingerprint():
    legacy, default = load(cfg.LEGACY_CONFIG), load()
    assert legacy.formatter == "percentages-v2" and legacy.data == LEGACY and legacy.data["strategy"] == cfg.FROZEN_STRATEGY
    assert legacy.fingerprint != default.fingerprint and len(legacy.fingerprint) == 64


def test_the_environment_and_the_cli_can_select_the_legacy_config():
    assert load(env={"MERALION_CONFIG": str(cfg.LEGACY_CONFIG)}).formatter == "percentages-v2"
    assert load(cli={"config": str(cfg.LEGACY_CONFIG)}).formatter == "percentages-v2"
    assert load(env={"MERALION_CONFIG": str(cfg.LEGACY_CONFIG)}, cli={"config": str(cfg.BUDGETED_CONFIG)}).formatter == "percentages-v2+amount-v1"      # the CLI wins


def test_the_formatter_names_the_config_accepts():
    assert cfg.FORMATTERS == ("none", "percentages-v2", "percentages-v2+amount-v1")
    for name in cfg.FORMATTERS:
        assert load(variant(formatter=name)).formatter == name
    with pytest.raises(cfg.ConfigError, match="unknown formatter"):
        load(variant(formatter="amount-v1"))                                                   # the amount step alone is not a selectable formatter


def test_the_default_formatter_change_alone_changes_the_fingerprint():
    default = load()
    assert load(variant(formatter="percentages-v2")).fingerprint != default.fingerprint
    assert load(variant(formatter="percentages-v2+amount-v1")).fingerprint == default.fingerprint
