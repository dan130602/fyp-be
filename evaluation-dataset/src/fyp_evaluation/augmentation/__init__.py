"""environment_v2: a derived, client-only audio-environment variant of the 250 final calls (offline; no model or paid API).

Each call has ONE condition at ONE fixed level: C1 clean, N1 office / N2 cafe / N3 train noise, D1 telephone, D2 muffled.
Only CLIENT turn regions are changed; every other sample is the G9-pinned source sample. Speech is never regenerated, resampled or
shifted: the output keeps the source sample rate and frame count, so the G9 timings stay valid. `assignments.json` depends only on the
frozen split and call metadata; `config.json` (levels, assets, DRAFT/FROZEN) depends on the assignments, never the other way round.
"""

VARIANT_ID = "environment_v2"
SEED = 20261112
ALGORITHM = {"name": "stratified-swap-annealing", "version": "1.1"}
CLEAN = "C1"
# condition -> (kind, severities). Every condition has a single level, so the severity tuple is empty and a call is described by its
# condition id alone (kept as a tuple so the severity column stays available if a second level is ever reinstated).
CONDITIONS = {
    "C1": ("clean", ()),
    "N1": ("noise", ()),
    "N2": ("noise", ()),
    "N3": ("noise", ()),
    "D1": ("telephone", ()),
    "D2": ("muffle", ()),
}
CONDITION_ORDER = tuple(CONDITIONS)
AUDIO_ASSETS = "audio_assets/noise"          # under the evaluation-dataset root
VARIANT_ROOT = "audio_augmentations/environment_v2"   # under the dataset root
CALL_VARIANT_DIR = "audio_variants/environment_v2"    # under calls/CALL_xxx


class AugmentationError(ValueError):
    """An input, contract or validation rule of environment_v2 is violated."""
