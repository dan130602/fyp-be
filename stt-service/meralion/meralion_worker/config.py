"""Runtime configuration: load, apply overrides, validate the EFFECTIVE configuration, fingerprint it.

Order: config file -> environment -> CLI -> validation. Only the swappable choices are part of the configuration and its fingerprint (config_id, strategy, decoder, formatter,
output_schema, model_manifest). Environment facts (model directory, region, output bucket) are carried next to it but are not part of its identity: the model's identity is its manifest.

settings_id is derived from the backend by a fixed table and cannot be set on its own, so backend and settings identity cannot disagree.
"""
import hashlib
import json
import os
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent.parent          # stt-service/meralion (in the image: /opt/meralion)
CONFIG_ROOT = PACKAGE_ROOT / "configs"
# The selected pipeline (Stage 2A): fg175-m2ax-seq-ap-lmonly-m5-fee2-amt1, formatter percentages-v2+amount-v1. The legacy configuration stays in the repository and is selectable with --config or
# MERALION_CONFIG; it differs only in config_id and formatter.
# Stage 3 follow-up B: the default is the same pipeline WITHOUT the invocation-wide retry budget (strategy v2), so a recording gets the same retries alone as in a batch. The two earlier
# configurations stay in the repository and are selectable: BUDGETED_CONFIG (amt1 with the 15% invocation budget) and LEGACY_CONFIG (fee2 formatter, same budget).
DEFAULT_CONFIG = CONFIG_ROOT / "runtime" / "fg175-m2ax-seq-ap-lmonly-m5-fee2-amt1-nobudget.json"
BUDGETED_CONFIG = CONFIG_ROOT / "runtime" / "fg175-m2ax-seq-ap-lmonly-m5-fee2-amt1.json"
LEGACY_CONFIG = CONFIG_ROOT / "runtime" / "fg175-m2ax-seq-ap-lmonly-m5-fee2.json"
CANDIDATE_RP105_CONFIG = CONFIG_ROOT / "runtime" / "fg175-m2ax-seq-ap-lmonly-m5-fee2-amt1-nobudget-vllm-rp105.json"          # the ONE vLLM candidate (Stage 3 follow-up D)
DEFAULT_REGION = "ap-southeast-1"                              # same default as evaluation-engine/transcribe_eval/config.py

SETTINGS_BY_BACKEND = {"transformers": "stock-transformers-v1", "vllm": "stock-vllm-package-v1"}          # a backend's STOCK settings: what a backend override resets to
ALLOWED_SETTINGS = {"transformers": ("stock-transformers-v1",),
                    "vllm": ("stock-vllm-package-v1", "vllm-rp105-v1", "stock-vllm-package-sched0-v1", "vllm-rp105-sched0-v1", "vllm-rp105-sched0-fp16-v1")}          # a candidate or a scheduling profile is selected only by naming it
FORMATTERS = ("none", "percentages-v2", "percentages-v2+amount-v1")
OUTPUT_SCHEMAS = ("meralion-worker-output-v0",)
CONFIG_KEYS = ("config_id", "strategy", "decoder", "formatter", "output_schema", "model_manifest")

# The one strategy v0 implements (handoff sections 1 and 6). Any other value is rejected.
FROZEN_STRATEGY = {
    "id": "fg-m2ax-seq-ap-lmonly-v1",
    "window_seconds": 30,
    "stride_seconds": 15,
    "merge_policy": "m2ax_seq_ap_lm",
    "retry_kinds": ["unresolved", "low_margin_conflict"],
    "unresolved_call_cap": 0.10,
    "run_cap": 0.15,
    "run_cap_scope": "invocation",
    "reconcile": {"conflict_tiebreak": False, "low_margin_tiebreak": True, "exclude_critical_from_low_margin": True},
}
# v2 (Stage 3 follow-up B): identical to v1 except that there is NO invocation-wide retry budget. The evaluated runs never hit that budget (612 against 502 retries on 175 calls, 137 against 90 on the
# pilot), so v2 reproduces what was measured while a recording gets the same retries alone as in a batch (v1 gave CALL_064 one retry alone and three in the 22-call invocation). A per-recording
# budget would NOT reproduce it (it would have removed 119 of the 502 retries). The unresolved per-call cap, the low-margin selection, the critical-content exclusion and every merge and
# reconcile rule are unchanged.
STRATEGY_V2 = {**FROZEN_STRATEGY, "id": "fg-m2ax-seq-ap-lmonly-v2", "run_cap": None, "run_cap_scope": "none"}
STRATEGIES = {FROZEN_STRATEGY["id"]: FROZEN_STRATEGY, STRATEGY_V2["id"]: STRATEGY_V2}


class ConfigError(ValueError):
    """The effective configuration is invalid (exit code 3)."""


class EffectiveConfig:
    def __init__(self, data, *, model_dir, region, output_bucket, config_path, config_root):
        self.data = data
        self.model_dir = Path(model_dir)
        self.region = region
        self.output_bucket = output_bucket
        self.config_path = config_path
        self.config_root = Path(config_root)

    @property
    def fingerprint(self):
        return fingerprint_of(self.data)

    @property
    def backend(self):
        return self.data["decoder"]["backend"]

    @property
    def formatter(self):
        return self.data["formatter"]

    @property
    def model_manifest_id(self):
        return self.data["model_manifest"]

    def as_record(self):
        """What is recorded with every result: the effective configuration and its fingerprint."""
        return {"config": json.loads(json.dumps(self.data)), "config_fingerprint": self.fingerprint}


def canonical(data):
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def fingerprint_of(data):
    return hashlib.sha256(canonical(data).encode("utf-8")).hexdigest()


def _read_file(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise ConfigError(f"config file not found: {path}")
    except json.JSONDecodeError as error:
        raise ConfigError(f"config file is not valid JSON: {path}: {error}")


def _validate(data, config_root):
    if not isinstance(data, dict):
        raise ConfigError("the configuration must be a JSON object")
    missing = [key for key in CONFIG_KEYS if key not in data]
    extra = [key for key in data if key not in CONFIG_KEYS]
    if missing or extra:
        raise ConfigError(f"configuration keys must be exactly {list(CONFIG_KEYS)} (missing {missing}, unexpected {extra})")
    known = STRATEGIES.get(data["strategy"].get("id")) if isinstance(data["strategy"], dict) else None
    if known is None or data["strategy"] != known:
        raise ConfigError(f"strategy differs from the frozen strategy of its id; the strategies this worker implements are exactly {sorted(STRATEGIES)}")
    decoder = data["decoder"]
    if not isinstance(decoder, dict) or set(decoder) != {"backend", "settings_id"}:
        raise ConfigError("decoder must be an object with exactly backend and settings_id")
    if decoder["backend"] not in SETTINGS_BY_BACKEND:
        raise ConfigError(f"unknown backend {decoder['backend']!r}; expected one of {sorted(SETTINGS_BY_BACKEND)}")
    if decoder["settings_id"] not in ALLOWED_SETTINGS[decoder["backend"]]:
        raise ConfigError(f"settings_id {decoder['settings_id']!r} contradicts backend {decoder['backend']!r} (allowed: {list(ALLOWED_SETTINGS[decoder['backend']])})")
    if data["formatter"] not in FORMATTERS:
        raise ConfigError(f"unknown formatter {data['formatter']!r}; expected one of {list(FORMATTERS)}")
    if data["output_schema"] not in OUTPUT_SCHEMAS:
        raise ConfigError(f"unknown output_schema {data['output_schema']!r}")
    manifest = Path(config_root) / "models" / f"{data['model_manifest']}.json"
    if not manifest.is_file():
        raise ConfigError(f"model manifest not found: {manifest}")


def load_runtime_config(path=None, *, env=None, cli=None, config_root=None):
    """Load, override and validate. `path` may be a file path or an already-parsed dict.

    cli: optional dict with keys config, backend, model_dir (CLI wins over the environment, which wins over the file).
    env: mapping (default os.environ): MERALION_CONFIG, MERALION_BACKEND, MERALION_MODEL_DIR, TRANSCRIBE_OUTPUT_BUCKET, AWS_REGION, AWS_DEFAULT_REGION.
    """
    env = os.environ if env is None else env
    cli = cli or {}
    config_root = Path(config_root) if config_root else CONFIG_ROOT
    if isinstance(path, dict):
        data, source = json.loads(json.dumps(path)), None
    else:
        chosen = path or cli.get("config") or env.get("MERALION_CONFIG") or DEFAULT_CONFIG
        data, source = _read_file(chosen), Path(chosen)
    _validate(data, config_root)                                  # the file as written must be consistent before any override (a contradictory pair is rejected, not repaired)
    backend = cli.get("backend") or env.get("MERALION_BACKEND")
    if backend:
        if backend not in SETTINGS_BY_BACKEND:
            raise ConfigError(f"unknown backend {backend!r}; expected one of {sorted(SETTINGS_BY_BACKEND)}")
        if data["decoder"]["backend"] != backend:                                                 # switching backend resets to the new backend's STOCK settings (never a candidate by accident);
            data["decoder"] = {"backend": backend, "settings_id": SETTINGS_BY_BACKEND[backend]}   # naming the file's own backend again keeps the settings the file names
    settings_id = cli.get("decoder_settings") or env.get("MERALION_DECODER_SETTINGS")            # file < environment < CLI; naming a settings id is the only way to select a candidate
    if settings_id:
        data["decoder"] = {"backend": data["decoder"]["backend"], "settings_id": settings_id}      # the pair is validated below: an id of another backend is "contradicts" (exit 3)
    _validate(data, config_root)                                  # and the effective configuration after the overrides
    model_dir = cli.get("model_dir") or env.get("MERALION_MODEL_DIR") or str(PACKAGE_ROOT / "model")
    region = env.get("AWS_REGION") or env.get("AWS_DEFAULT_REGION") or DEFAULT_REGION
    return EffectiveConfig(data, model_dir=model_dir, region=region, output_bucket=env.get("TRANSCRIBE_OUTPUT_BUCKET") or None, config_path=source, config_root=config_root)
