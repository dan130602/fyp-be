"""Command line: run, check-model, verify-environment.

Exit codes: 0 ok / every member succeeded;  1 runtime abort (resumable: run again with the same --invocation-id);  2 at least one final input failure after the others were processed;
3 configuration, model, environment, decoding-settings or resume-compatibility failure. The last stdout line is always a JSON summary.
"""
import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path

from meralion_worker import __version__
from meralion_worker import config as config_module
from meralion_worker import model_check, output, pipeline
from meralion_worker.audio import load_recording
from meralion_worker.decoders.base import DecodingSettingsError
from meralion_worker.state import INVOCATION_ID, ResumeError
from meralion_worker.storage import LocalStorage, S3Storage, parse_s3_uri

EXIT_OK, EXIT_RUNTIME, EXIT_INPUT, EXIT_CONFIG = 0, 1, 2, 3
CONFIG_ERRORS = (config_module.ConfigError, model_check.CheckError, DecodingSettingsError, ResumeError)


class UsageError(ValueError):
    """A bad command line or jobs file (exit code 3)."""


def _emit(summary, summary_output=None):
    line = json.dumps(summary, ensure_ascii=False, sort_keys=True)
    print(line)
    if summary_output:
        Path(summary_output).write_text(line + "\n", encoding="utf-8")


def _cli_overrides(args):
    return {"config": args.config, "backend": getattr(args, "backend", None), "model_dir": getattr(args, "model_dir", None), "decoder_settings": getattr(args, "decoder_settings", None)}


# ---- run ------------------------------------------------------------------------------------------------------------------------------------------------------------

def run_checks(effective):
    """The identity checks that precede any work. Module-level so tests can replace it; the command line has no flag that skips it."""
    manifest = model_check.load_model_manifest(effective.model_manifest_id, effective.config_root)
    model_report = model_check.check_model(effective.model_dir, manifest)
    vendored_report = model_check.verify_vendored()
    package_manifest = model_check.load_json(effective.config_root / "third_party" / f"{model_check.PACKAGE_MANIFEST_ID}.json")
    package_report = model_check.verify_package(package_manifest)
    return {"manifest": manifest, "model": model_report, "vendored": vendored_report, "package": package_report}


def make_decoder(effective):
    """Build the configured decoder; it verifies the frozen decoding settings before returning. Module-level so tests can replace it."""
    if effective.backend == "transformers":
        from meralion_worker.decoders.transformers_decoder import TransformersDecoder
        return TransformersDecoder(effective.model_dir)
    from meralion_worker.decoders.vllm_decoder import VllmDecoder
    return VllmDecoder(effective.model_dir, settings_id=effective.data["decoder"]["settings_id"])


def parse_jobs(args):
    if args.jobs:
        if args.job_name or args.input:
            raise UsageError("give --jobs or --job-name with --input, not both")
        try:
            data = json.loads(Path(args.jobs).read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            raise UsageError(f"cannot read the jobs file: {error}")
        entries = data.get("jobs") if isinstance(data, dict) else None
        if not isinstance(entries, list) or not entries:
            raise UsageError('the jobs file must be {"jobs": [{"job_name": ..., "input": ...}, ...]} with at least one job')
    else:
        if not (args.job_name and args.input):
            raise UsageError("give --jobs, or both --job-name and --input")
        entries = [{"job_name": args.job_name, "input": args.input}]
    jobs = []
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != {"job_name", "input"} or not all(isinstance(entry[k], str) and entry[k] for k in entry):
            raise UsageError('each job must be {"job_name": <string>, "input": <string>}')
        if not output.JOB_NAME.fullmatch(entry["job_name"]):
            raise UsageError(f"invalid job name {entry['job_name']!r}: it must fully match {output.JOB_NAME.pattern}")
        jobs.append(pipeline.Job(entry["job_name"], entry["input"]))
    names = [job.job_name for job in jobs]
    if len(set(names)) != len(names):
        raise UsageError("job names must be unique")
    return jobs


def build_storages(args, effective):
    local = bool(args.output_dir or args.state_dir)
    remote = bool(args.output_bucket or args.state_uri)
    if local and remote:
        raise UsageError("use either --output-dir with --state-dir, or --output-bucket with --state-uri; not a mix")
    if local:
        if not (args.output_dir and args.state_dir):
            raise UsageError("--output-dir needs --state-dir (and the other way round)")
        return LocalStorage(args.output_dir), LocalStorage(args.state_dir)
    bucket = args.output_bucket or effective.output_bucket
    if not (bucket and args.state_uri):
        raise UsageError("give --output-dir with --state-dir, or --output-bucket (or TRANSCRIBE_OUTPUT_BUCKET) with --state-uri")
    state_bucket, state_prefix = parse_s3_uri(args.state_uri)
    return S3Storage(bucket, region=effective.region), S3Storage(state_bucket, state_prefix, region=effective.region)


def make_open_recording(effective):
    """A callable job -> recording. An s3:// input is downloaded for each load (so a changed object is noticed by its hash) and deleted after it was read."""
    def open_recording(job):
        if job.input.startswith("s3://"):
            with tempfile.TemporaryDirectory() as scratch:
                path = S3Storage(parse_s3_uri(job.input)[0], region=effective.region).download(job.input, scratch)
                return load_recording(job.job_name, path)
        return load_recording(job.job_name, LocalStorage(".").download(job.input, None))
    return open_recording


def command_run(args):
    if not INVOCATION_ID.fullmatch(args.invocation_id):
        raise UsageError(f"--invocation-id must fully match {INVOCATION_ID.pattern}")
    jobs = parse_jobs(args)
    effective = config_module.load_runtime_config(cli=_cli_overrides(args))
    output_storage, state_storage = build_storages(args, effective)
    checks = run_checks(effective)
    started = time.monotonic()
    decoder = make_decoder(effective)
    load_seconds = round(time.monotonic() - started, 3)
    context = pipeline.RunContext(
        model={"manifest": checks["manifest"]["id"], "manifest_sha256": checks["model"]["manifest_sha256"], "revision": checks["manifest"]["revision"]},
        worker={"version": __version__, "git_commit": os.environ.get("MERALION_WORKER_GIT_COMMIT", "unknown"), "vendored": checks["vendored"]},
        vendored_sources_sha256=checks["vendored"]["sources_sha256"], model_manifest_sha256=checks["model"]["manifest_sha256"],
        timing={"model_check_s": checks["model"]["seconds"], "model_load_s": load_seconds})
    summary = pipeline.run_invocation(
        invocation_id=args.invocation_id, effective=effective, jobs=jobs, decoder=decoder, open_recording=make_open_recording(effective),
        output_storage=output_storage, state_storage=state_storage, context=context, log=lambda message: print(message, file=sys.stderr, flush=True))
    _emit({"command": "run", "ok": summary.exit_code == EXIT_OK, "config_fingerprint": effective.fingerprint, **summary.as_dict()}, args.summary_output)
    return summary.exit_code


# ---- check-model / verify-environment ------------------------------------------------------------------------------------------------------------------------

def command_check_model(args):
    effective = config_module.load_runtime_config(cli=_cli_overrides(args))
    manifest = model_check.load_model_manifest(effective.model_manifest_id, effective.config_root)
    report = model_check.check_model(effective.model_dir, manifest)
    _emit({"command": "check-model", "ok": True, "model_manifest": manifest["id"], **report}, args.summary_output)
    return EXIT_OK


def command_verify_environment(args):
    effective = config_module.load_runtime_config(cli=_cli_overrides(args))
    vendored_report = model_check.verify_vendored()
    package_manifest = model_check.load_json(effective.config_root / "third_party" / f"{model_check.PACKAGE_MANIFEST_ID}.json")
    package_report = model_check.verify_package(package_manifest)
    _emit({"command": "verify-environment", "ok": True, "worker_version": __version__, "config_fingerprint": effective.fingerprint, "vendored": vendored_report, "package": package_report}, args.summary_output)
    return EXIT_OK


def build_parser():
    parser = argparse.ArgumentParser(prog="meralion_worker", description="MERaLiON STT worker")
    parser.add_argument("--version", action="version", version=f"meralion_worker {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)

    def common(sub):
        sub.add_argument("--config", help="runtime config file (default: MERALION_CONFIG, then the packaged frozen config)")
        sub.add_argument("--backend", choices=sorted(config_module.SETTINGS_BY_BACKEND))
        sub.add_argument("--decoder-settings", dest="decoder_settings", help="a decoder settings id (a stock id or a named candidate; default: the config's; MERALION_DECODER_SETTINGS): the only way to select a candidate")
        sub.add_argument("--model-dir", dest="model_dir", help="model directory (default: MERALION_MODEL_DIR, then ./model)")
        sub.add_argument("--summary-output", help="also write the JSON summary to this file")

    run = commands.add_parser("run", help="transcribe the recordings of one invocation (resumable with the same --invocation-id)")
    common(run)
    run.add_argument("--invocation-id", required=True, help="names the invocation and its state; the same id resumes it")
    run.add_argument("--jobs", help='JSON file {"jobs": [{"job_name": ..., "input": <path or s3://bucket/key>}]}')
    run.add_argument("--job-name")
    run.add_argument("--input")
    run.add_argument("--output-dir", help="local mode: transcripts/<job>.json is written here (with --state-dir)")
    run.add_argument("--state-dir", help="local mode: progress state")
    run.add_argument("--output-bucket", help="S3 mode: transcripts/<job>.json goes to this bucket (default: TRANSCRIBE_OUTPUT_BUCKET; with --state-uri)")
    run.add_argument("--state-uri", help="S3 mode: s3://bucket/prefix for the progress state")
    run.set_defaults(handler=command_run)

    for name, handler, help_text in (("check-model", command_check_model, "hash the model directory against its manifest (refuses a pending manifest)"),
                                     ("verify-environment", command_verify_environment, "check the vendored sources and the installed MERaLiON package (refuses a pending manifest)")):
        sub = commands.add_parser(name, help=help_text)
        common(sub)
        sub.set_defaults(handler=handler)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        return args.handler(args)
    except (UsageError, *CONFIG_ERRORS) as error:
        _emit({"command": args.command, "ok": False, "error": str(error), "exit_code": EXIT_CONFIG}, getattr(args, "summary_output", None))
        return EXIT_CONFIG
    except Exception as error:                                       # noqa: BLE001  a runtime failure of `run` is resumable: report it and exit 1
        if args.command != "run":
            raise
        _emit({"command": "run", "ok": False, "error": f"{type(error).__name__}: {error}", "exit_code": EXIT_RUNTIME, "invocation_id": args.invocation_id}, args.summary_output)
        return EXIT_RUNTIME


if __name__ == "__main__":
    sys.exit(main())
