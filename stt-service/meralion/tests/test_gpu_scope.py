"""REV 4.1 A3: a GPU-memory number is a SLICE measurement only when its device IDENTITY is established (the returned UUID is the allocated one) and its capacity fits the slice; nearby numbers prove nothing."""
import importlib.util
import json
import types
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parent.parent / "tools"
spec = importlib.util.spec_from_file_location("gpu_scope_under_test", TOOLS / "gpu_scope.py")
gs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gs)

SLICE, PARENT, OTHER, WHOLE = "MIG-aaaaaaaa-1111", "GPU-bbbbbbbb-2222", "MIG-cccccccc-3333", "GPU-dddddddd-4444"
HEADER = "timestamp, name, uuid, memory.used [MiB], memory.total [MiB], utilization.gpu [%]"
PREAMBLE = "# allocated device: CUDA_VISIBLE_DEVICES=MIG-aaaaaaaa-1111\nGPU 0: NVIDIA RTX PRO 6000 (UUID: GPU-bbbbbbbb-2222)\n  MIG 1g.24gb Device 0: (UUID: MIG-aaaaaaaa-1111)\n"


def sampler(rows, preamble=PREAMBLE, header=HEADER):
    return preamble + header + chr(10) + chr(10).join(rows) + chr(10)


def row(uuid=SLICE, used=7900, total=24000, name="NVIDIA RTX PRO 6000"):
    return f"2026/11/17 10:00:00.000, {name}, {uuid}, {used} MiB, {total} MiB, [N/A]"


def parsed(rows, **kw):
    return gs.parse_sampler(sampler(rows, **kw))


# ---- the sampler file ---------------------------------------------------------------------------------------------------------------------------------------------------------------------

def test_a_valid_sampler_file_is_parsed_and_the_preamble_is_ignored():
    result = parsed([row(used=254), row(used=7918)])
    assert [r["used_mib"] for r in result["rows"]] == [254, 7918] and result["malformed"] == 0 and result["not_available"] == 0
    assert {r["uuid"] for r in result["rows"]} == {SLICE} and gs.raw_peak_mib is not None


def test_error_text_not_available_values_and_truncated_lines_are_counted_never_turned_into_numbers():
    text = sampler([row(), "Failed to parse device handle: MIG-aaaaaaaa-1111", "2026/11/17 10:00:05.000, NVIDIA, MIG-aaaaaaaa-1111, [N/A], [N/A], [N/A]", "2026/11/17 10:00:10.000, NVIDIA, MIG-aaaaaaaa-1111, 12"])
    result = gs.parse_sampler(text)
    assert len(result["rows"]) == 1 and result["malformed"] == 2 and result["not_available"] == 1


def test_a_value_that_is_not_a_memory_figure_is_malformed_not_a_row_and_not_a_missing_value():
    text = sampler([row(), "2026/11/17 10:00:05.000, NVIDIA, MIG-aaaaaaaa-1111, lots MiB, 24000 MiB, 0 %", "2026/11/17 10:00:10.000, NVIDIA, MIG-aaaaaaaa-1111, 12 MiB, [N/A], 0 %"])
    result = gs.parse_sampler(text)
    assert len(result["rows"]) == 1 and result["malformed"] == 1 and result["not_available"] == 1


def test_a_header_only_file_has_no_rows_and_the_scope_is_unavailable():
    result = parsed([])
    assert result["rows"] == []
    verdict = gs.evaluate(allocated=SLICE, parent=PARENT, parsed=result, parent_total_mib=97887)
    assert verdict["scope"] == "unavailable" and verdict["reason"] == "no_valid_rows"


def test_only_not_available_values_give_unavailable_with_its_own_reason():
    verdict = gs.evaluate(allocated=SLICE, parent=PARENT, parsed=gs.parse_sampler(sampler(["2026/11/17 10:00:00.000, NVIDIA, MIG-aaaaaaaa-1111, [N/A], [N/A], [N/A]"])), parent_total_mib=97887)
    assert verdict["scope"] == "unavailable" and verdict["reason"] == "memory_not_available"


# ---- identity first, then capacity, then the numbers ----------------------------------------------------------------------------------------------------------------------------------------

def verdict_of(rows, **kw):
    arguments = dict(allocated=SLICE, parent=PARENT, parent_total_mib=97887)
    arguments.update(kw)
    return gs.evaluate(parsed=parsed(rows), **arguments)


def test_the_allocated_slice_with_a_slice_sized_capacity_is_verified():
    result = verdict_of([row(), row(used=8100)])
    assert result["scope"] == "slice_verified" and result["reason"] is None and result["device_kind"] == "mig_slice" and result["memory_total_mib"] == 24000


def test_the_parent_card_with_coincidentally_similar_usage_is_not_the_slice():
    """Numbers that agree with torch's do not make the PARENT uuid the slice."""
    result = verdict_of([row(uuid=PARENT, used=7900, total=97887)], numbers={"measured_used_mib": 7900, "torch_used_mib": 7900})
    assert result["scope"] == "scope_unverified" and result["reason"] == "parent_identity"


def test_the_right_uuid_with_a_parent_sized_capacity_is_not_a_slice_measurement():
    result = verdict_of([row(total=97887)])
    assert result["scope"] == "scope_unverified" and result["reason"] == "capacity_parent_sized"
    assert verdict_of([row(total=90000)])["reason"] == "capacity_parent_sized"


@pytest.mark.parametrize("uuid_cell", ["", "[N/A]", "[Not Supported]", "not-a-uuid"])
def test_a_missing_empty_or_unusable_uuid_never_verifies(uuid_cell):
    result = verdict_of([row(uuid=uuid_cell)])
    assert result["scope"] == "scope_unverified" and result["reason"] == "identity_missing"


def test_a_header_without_a_uuid_column_cannot_verify_a_slice():
    text = "timestamp, name, memory.used [MiB], memory.total [MiB], utilization.gpu [%]" + chr(10) + "2026/11/17 10:00:00.000, NVIDIA, 7900 MiB, 24000 MiB, [N/A]" + chr(10)
    result = gs.evaluate(allocated=SLICE, parent=PARENT, parsed=gs.parse_sampler(text), parent_total_mib=97887)
    assert result["scope"] == "scope_unverified" and result["reason"] == "identity_missing"


def test_another_slices_uuid_is_a_mismatch_and_one_stray_row_spoils_the_run():
    assert verdict_of([row(uuid=OTHER)])["reason"] == "identity_mismatch"
    assert verdict_of([row(), row(uuid=OTHER)])["reason"] == "identity_mismatch"
    assert verdict_of([row(), row(uuid="")])["reason"] == "identity_missing"


def test_agreeing_numbers_never_upgrade_a_wrong_identity_and_disagreeing_numbers_downgrade_a_right_one():
    wrong = verdict_of([row(uuid=OTHER, used=7900)], numbers={"measured_used_mib": 7900, "torch_used_mib": 7900})
    assert wrong["scope"] == "scope_unverified"
    agree = verdict_of([row(used=7900)], numbers={"measured_used_mib": 7900, "torch_used_mib": 8100})
    assert agree["scope"] == "slice_verified"
    disagree = verdict_of([row(used=7900)], numbers={"measured_used_mib": 7900, "torch_used_mib": 30000})
    assert disagree["scope"] == "scope_unverified" and disagree["reason"] == "numbers_disagree"
    edge = verdict_of([row(used=10000)], numbers={"measured_used_mib": 10000, "torch_used_mib": 10256})
    assert edge["scope"] == "slice_verified"                                                            # exactly within max(256 MiB, 3 %)


def test_the_capacity_must_agree_with_torch_when_torch_gives_one_and_the_parent_capacity_must_be_known():
    assert verdict_of([row(total=24000)], torch_total_mib=24000)["scope"] == "slice_verified"
    assert verdict_of([row(total=24000)], torch_total_mib=20000)["reason"] == "capacity_inconsistent"
    assert verdict_of([row(total=24000)], parent_total_mib=None)["reason"] == "parent_capacity_unknown"
    assert verdict_of([row(total=24000), row(total=23000)])["reason"] == "capacity_inconsistent"


def test_a_whole_gpu_is_device_verified_by_identity_alone():
    result = gs.evaluate(allocated=WHOLE, parent=WHOLE, parsed=parsed([row(uuid=WHOLE, total=46068)]))
    assert result["scope"] == "device_verified" and result["device_kind"] == "whole_gpu"
    assert gs.evaluate(allocated=WHOLE, parent=WHOLE, parsed=parsed([row(uuid=OTHER, total=46068)]))["scope"] == "scope_unverified"


def test_an_unknown_allocated_device_cannot_be_verified():
    assert gs.evaluate(allocated=None, parent=None, parsed=parsed([row()]))["reason"] == "identity_missing"
    assert gs.evaluate(allocated="0", parent=None, parsed=parsed([row()]))["reason"] == "identity_missing"


# ---- the NVML method ------------------------------------------------------------------------------------------------------------------------------------------------------------------------

class FakeNvml:
    def __init__(self, uuid=SLICE, total=24000, mig=True, parent_uuid=PARENT, parent_total=97887, fail=None):
        self.fail, self.state = fail, (uuid, total, mig, parent_uuid, parent_total)

    def nvmlInit(self):
        if self.fail == "init":
            raise RuntimeError("NVML Shared Library Not Found")

    def nvmlDeviceGetHandleByUUID(self, uuid):
        if self.fail == "handle":
            raise RuntimeError("Not Found")
        return ("slice", uuid)

    def nvmlDeviceGetMemoryInfo(self, handle):
        total = self.state[1] if handle[0] == "slice" else self.state[4]
        return types.SimpleNamespace(total=total * 2 ** 20, used=7900 * 2 ** 20)

    def nvmlDeviceGetUUID(self, handle):
        return (self.state[0] if handle[0] == "slice" else self.state[3]).encode()

    def nvmlDeviceIsMigDeviceHandle(self, handle):
        return 1 if self.state[2] else 0

    def nvmlDeviceGetDeviceHandleFromMigDeviceHandle(self, handle):
        return ("parent", self.state[3])


def test_the_nvml_reading_is_verified_only_for_a_mig_handle_with_the_allocated_uuid_and_a_slice_sized_total():
    probe = gs.nvml_probe(SLICE, FakeNvml())
    assert probe["available"] and probe["is_mig_handle"] is True and probe["parent_uuid"] == PARENT and probe["parent_total_mib"] == 97887
    assert gs.evaluate_nvml(SLICE, probe)["scope"] == "slice_verified"
    assert gs.evaluate_nvml(SLICE, gs.nvml_probe(SLICE, FakeNvml(uuid=PARENT, total=97887)))["reason"] in ("parent_identity", "identity_mismatch")
    assert gs.evaluate_nvml(SLICE, gs.nvml_probe(SLICE, FakeNvml(total=97887)))["reason"] == "capacity_parent_sized"
    assert gs.evaluate_nvml(SLICE, gs.nvml_probe(SLICE, FakeNvml(mig=False)))["reason"] == "not_a_mig_handle"


@pytest.mark.parametrize("fail", ["init", "handle"])
def test_an_nvml_failure_is_recorded_not_raised(fail):
    probe = gs.nvml_probe(SLICE, FakeNvml(fail=fail))
    assert probe["available"] is False and probe["error"] and gs.evaluate_nvml(SLICE, probe) == {"scope": "unavailable", "reason": "nvml_unavailable"}


# ---- the record of one run, and how a report may use it -------------------------------------------------------------------------------------------------------------------------------------

def make_run(tmp_path, rows, *, allocated=SLICE, parent=PARENT, record_overrides=None, meta_gpu=None, write_record=True, parent_total="97887"):
    folder = tmp_path / "run"
    folder.mkdir()
    csv_path = folder / "gpu.csv"
    csv_path.write_text(sampler(rows), encoding="utf-8")
    meta = {"gpu_uuid": meta_gpu or allocated, "parent_gpu_uuid": parent, "slurm_job_id": "55"}
    (folder / "run-meta.json").write_text(json.dumps(meta), encoding="utf-8")

    def fake_run(command, **kwargs):
        text = parent_total if "memory.total" in " ".join(command) else "590.48.01"
        return types.SimpleNamespace(returncode=0, stdout=text + chr(10), stderr="")
    if write_record:
        record = gs.build_record(meta, csv_path, run=fake_run, nvml=FakeNvml(), env={})
        record.update(record_overrides or {})
        (folder / gs.SCOPE_FILE).write_text(json.dumps(record), encoding="utf-8")
    return csv_path


def test_a_verified_bound_record_gives_a_number_and_a_label_that_says_what_it_is(tmp_path):
    report = gs.memory_report(make_run(tmp_path, [row(used=254), row(used=7918)]))
    assert report["scope"] == "slice_verified" and report["peak_mib"] == 7918 and report["device_reported_peak_mib"] == 7918
    assert report["method"] == gs.CSV_METHOD and report["device_uuid"] == SLICE and "slice memory" in report["label"] and "reservation" in report["label"]


def test_a_well_formed_sampler_file_with_a_failed_scope_record_is_not_a_slice_measurement(tmp_path):
    """The valid CSV must not bypass the failed scope check (Codex 1): the number is withheld, the raw figure is labelled."""
    report = gs.memory_report(make_run(tmp_path, [row(uuid=PARENT, used=7900, total=97887)]))
    assert report["scope"] == "scope_unverified" and report["reason"] == "parent_identity"
    assert report["peak_mib"] is None and report["device_reported_peak_mib"] == 7900 and "NOT verified" in report["label"] and "not a slice measurement" in report["label"]


def test_a_slice_uuid_with_parent_sized_capacity_withholds_the_number(tmp_path):
    report = gs.memory_report(make_run(tmp_path, [row(total=97887)]))
    assert report["peak_mib"] is None and report["scope"] == "scope_unverified" and report["reason"] == "capacity_parent_sized" and report["device_reported_peak_mib"] == 7900


def test_a_missing_uuid_withholds_the_number(tmp_path):
    report = gs.memory_report(make_run(tmp_path, [row(uuid="")]))
    assert report["peak_mib"] is None and report["reason"] == "identity_missing" and report["device_reported_peak_mib"] == 7900


def test_a_run_without_a_scope_record_has_no_slice_number(tmp_path):
    report = gs.memory_report(make_run(tmp_path, [row()], write_record=False))
    assert report["scope"] == "unavailable" and report["peak_mib"] is None and report["device_reported_peak_mib"] == 7900 and "before REV 4.1" in report["reason"]


def test_a_record_that_is_not_bound_to_this_run_is_unavailable(tmp_path):
    csv_path = make_run(tmp_path, [row()])
    record_path = csv_path.parent / gs.SCOPE_FILE
    original = json.loads(record_path.read_text(encoding="utf-8"))
    assert gs.memory_report(csv_path)["peak_mib"] == 7900
    for label, change, fragment in (("another device", {"allocated_uuid": OTHER}, "this run was allocated"), ("another method", {"method": "nvml-by-uuid"}, "the sampler used"),
                                    ("another file", {"sampler_csv_sha256": "0" * 64}, "changed after")):
        record_path.write_text(json.dumps({**original, **change}), encoding="utf-8")
        report = gs.memory_report(csv_path)
        assert report["scope"] == "unavailable" and report["peak_mib"] is None and fragment in report["reason"], label
    record_path.write_text(json.dumps(original), encoding="utf-8")
    csv_path.write_text(csv_path.read_text(encoding="utf-8") + row(used=99999) + chr(10), encoding="utf-8")        # the sampler file changed after the record was made
    assert gs.memory_report(csv_path)["peak_mib"] is None


def test_a_record_without_run_meta_cannot_be_bound_and_gives_no_number(tmp_path):
    csv_path = make_run(tmp_path, [row()])
    (csv_path.parent / "run-meta.json").unlink()
    report = gs.memory_report(csv_path)
    assert report["scope"] == "unavailable" and report["peak_mib"] is None and "run-meta records no allocated device" in report["reason"] and report["device_reported_peak_mib"] == 7900
    (csv_path.parent / "run-meta.json").write_text(json.dumps({"slurm_job_id": "55"}), encoding="utf-8")            # run-meta without the allocated device
    assert gs.memory_report(csv_path)["peak_mib"] is None


def test_the_smokes_slice_verdict_is_never_reused_for_another_runs_device(tmp_path):
    """Each job has its own slice UUID: a record made for one allocation says nothing about a run allocated another one."""
    csv_path = make_run(tmp_path, [row()], meta_gpu=OTHER)                       # the record was made for SLICE, run-meta says this run got OTHER
    meta = json.loads((csv_path.parent / "run-meta.json").read_text(encoding="utf-8"))
    record = json.loads((csv_path.parent / gs.SCOPE_FILE).read_text(encoding="utf-8"))
    assert meta["gpu_uuid"] == OTHER and record["allocated_uuid"] == OTHER           # build_record read OTHER from run-meta: the sampler's SLICE rows are a mismatch
    report = gs.memory_report(csv_path)
    assert report["scope"] == "scope_unverified" and report["reason"] == "identity_mismatch" and report["peak_mib"] is None


def test_an_old_whole_gpu_folder_and_a_missing_file_are_unavailable_not_zero(tmp_path):
    assert gs.memory_report(tmp_path / "nope.csv")["peak_mib"] is None and gs.memory_report(None)["scope"] == "unavailable"
    old = tmp_path / "old.csv"
    old.write_text("timestamp, name, memory.used [MiB]" + chr(10) + "2026/10/01 10:00:00.000, NVIDIA L40S, 3000 MiB" + chr(10), encoding="utf-8")
    assert gs.raw_peak_mib(old) == 3000 and gs.memory_report(old)["peak_mib"] is None


def test_the_record_carries_everything_needed_to_audit_it(tmp_path):
    csv_path = make_run(tmp_path, [row()])
    record = json.loads((csv_path.parent / gs.SCOPE_FILE).read_text(encoding="utf-8"))
    for key in ("allocated_uuid", "parent_uuid", "returned_uuids", "method", "driver", "node", "slurm_job_id", "memory_total_mib", "parent_total_mib", "sampler_csv_sha256", "scope", "reason", "nvml", "sampler_format"):
        assert key in record, key
    assert record["parent_total_mib"] == 97887 and record["nvml"]["verdict"]["scope"] == "slice_verified" and record["sampler_format"]["valid_rows"] == 1


def test_the_command_writes_the_record_and_never_fails_the_run(tmp_path, monkeypatch, capsys):
    csv_path = make_run(tmp_path, [row()], write_record=False)
    out = csv_path.parent / "scope.json"
    assert gs.main(["--run-meta", str(csv_path.parent / "run-meta.json"), "--csv", str(csv_path), "--out", str(out)]) == 0
    written = json.loads(out.read_text(encoding="utf-8"))
    assert written["scope"] == "scope_unverified" and written["reason"] == "parent_capacity_unknown" and written["nvml"]["verdict"]["scope"] == "unavailable"       # the failures are facts in the record
    assert gs.main(["--run-meta", str(tmp_path / "missing.json"), "--csv", str(csv_path), "--out", str(tmp_path / "x.json")]) == 0 and not (tmp_path / "x.json").exists()


# ---- the smoke's cross-check ---------------------------------------------------------------------------------------------------------------------------------------------------------------------

def test_the_smoke_scope_truth_table():
    shot = lambda uuid=SLICE, used=7900, total=24000: HEADER + chr(10) + row(uuid=uuid, used=used, total=total) + chr(10)
    base = dict(allocated=SLICE, parent=PARENT, torch_total_mib=24000, torch_used_mib=7950, parent_total_mib=97887)
    assert gs.smoke_scope(one_shot=shot(), **base)["gpu_memory_scope"] == "slice_verified"
    assert gs.smoke_scope(one_shot=shot(used=20000), **base)["reason"] == "numbers_disagree"
    assert gs.smoke_scope(one_shot=shot(uuid=PARENT, total=97887), **base)["reason"] == "parent_identity"
    assert gs.smoke_scope(one_shot=shot(total=97887), **base)["reason"] == "capacity_parent_sized"
    assert gs.smoke_scope(one_shot="", **base)["gpu_memory_scope"] == "unavailable"
    assert gs.smoke_scope(one_shot=None, **base)["gpu_memory_scope"] == "unavailable"


def test_the_tool_uses_the_standard_library_only_and_never_writes_the_device_variable():
    import ast
    source = (TOOLS / "gpu_scope.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported = {alias.name.split(".")[0] for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names}
    imported |= {node.module.split(".")[0] for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module}
    assert imported <= {"argparse", "csv", "hashlib", "json", "os", "platform", "re", "subprocess", "sys", "time", "pathlib", "pynvml"}
    assert "environ[" not in source and "putenv" not in source and "os.environ.pop" not in source        # the device variable is never written, unset or rewritten here


# ---- REV 4.2 (Codex round 2): the label is never copied, impossible values are never verified, malformed metadata never crashes ------------------------------------------------------------

def rewrite_record(csv_path, **changes):
    path = csv_path.parent / gs.SCOPE_FILE
    path.write_text(json.dumps({**json.loads(path.read_text(encoding="utf-8")), **changes}), encoding="utf-8")


def test_a_contradictory_scope_label_cannot_promote_the_parent_card_rows_to_a_slice_measurement(tmp_path):
    """Codex 1: rows carrying the PARENT uuid and the parent total, a record whose label was edited to slice_verified (hash, device and method all still consistent)."""
    csv_path = make_run(tmp_path, [row(uuid=PARENT, used=7900, total=97887)])
    rewrite_record(csv_path, scope="slice_verified", reason=None)
    report = gs.memory_report(csv_path)
    assert report["peak_mib"] is None and report["scope"] == "scope_unverified" and report["reason"].startswith("record_contradicts_sampler")
    assert "scope: record 'slice_verified'" in report["reason"] and report["device_reported_peak_mib"] == 7900 and "not a slice measurement" in report["label"]


def test_the_label_edit_is_caught_for_every_kind_of_wrong_evidence_not_only_the_parent_uuid(tmp_path):
    for index, rows in enumerate(([row(total=97887)], [row(uuid="")], [row(uuid=OTHER)])):               # parent-sized capacity, no identity, another device
        run = tmp_path / f"case{index}"
        run.mkdir()
        csv_path = make_run(run, rows)
        rewrite_record(csv_path, scope="slice_verified", reason=None)
        report = gs.memory_report(csv_path)
        assert report["peak_mib"] is None and report["reason"].startswith("record_contradicts_sampler"), index


def test_a_recorded_summary_that_disagrees_with_the_sampler_is_a_contradiction(tmp_path):
    for index, change in enumerate(({"memory_total_mib": 12345}, {"returned_uuids": [PARENT]}, {"reason": "something_else"}, {"scope": "scope_unverified", "reason": "capacity_parent_sized"})):
        run = tmp_path / f"c{index}"
        run.mkdir()
        csv_path = make_run(run, [row()])
        rewrite_record(csv_path, **change)
        report = gs.memory_report(csv_path)
        assert report["peak_mib"] is None and report["scope"] == "scope_unverified" and report["reason"].startswith("record_contradicts_sampler"), change


def test_hiding_the_torch_capacity_or_the_parent_capacity_in_the_record_is_a_contradiction_when_the_label_is_kept(tmp_path):
    csv_path = make_run(tmp_path, [row(total=24000)])
    record = json.loads((csv_path.parent / gs.SCOPE_FILE).read_text(encoding="utf-8"))
    assert record["torch_total_mib"] is None and gs.memory_report(csv_path)["scope"] == "slice_verified"
    rewrite_record(csv_path, torch_total_mib=40000)                                                      # a torch figure far from 24000: the derivation now says capacity_inconsistent, the label says verified
    assert gs.memory_report(csv_path)["reason"].startswith("record_contradicts_sampler")
    rewrite_record(csv_path, torch_total_mib=None, parent_total_mib=24000)                               # a parent capacity equal to the slice total: the derivation says parent-sized
    assert gs.memory_report(csv_path)["peak_mib"] is None


def test_a_record_written_without_the_torch_field_is_still_read(tmp_path):
    csv_path = make_run(tmp_path, [row()])
    path = csv_path.parent / gs.SCOPE_FILE
    record = json.loads(path.read_text(encoding="utf-8"))
    del record["torch_total_mib"]
    path.write_text(json.dumps(record), encoding="utf-8")
    assert gs.memory_report(csv_path)["peak_mib"] == 7900


def test_a_row_that_uses_more_memory_than_exists_is_impossible_and_never_verified(tmp_path):
    """Codex 2: used 30000 MiB of 24000 MiB on the allocated device."""
    result = parsed([row(used=30000, total=24000)])
    assert result["rows"] == [] and result["impossible"] == 1
    verdict = gs.evaluate(allocated=SLICE, parent=PARENT, parsed=result, parent_total_mib=97887)
    assert verdict["scope"] == "scope_unverified" and verdict["reason"] == "memory_used_exceeds_total"
    report = gs.memory_report(make_run(tmp_path, [row(used=30000, total=24000)]))
    assert report["peak_mib"] is None and report["scope"] == "scope_unverified" and report["reason"] == "memory_used_exceeds_total" and report["device_reported_peak_mib"] is None


def test_one_impossible_row_among_valid_ones_makes_the_stream_unverified_and_the_impossible_figure_is_never_reported(tmp_path):
    csv_path = make_run(tmp_path, [row(used=7900), row(used=30000, total=24000), row(used=8000)])
    report = gs.memory_report(csv_path)
    assert report["peak_mib"] is None and report["reason"] == "memory_used_exceeds_total" and report["device_reported_peak_mib"] == 8000
    assert gs.raw_peak_mib(csv_path) == 8000


def test_used_equal_to_total_is_possible_and_a_zero_total_is_not():
    assert parsed([row(used=24000, total=24000)])["impossible"] == 0
    zero = parsed([row(used=0, total=0)])
    assert zero["impossible"] == 1 and zero["rows"] == []


def test_the_nvml_reading_is_rejected_for_impossible_or_missing_memory_numbers():
    probe = gs.nvml_probe(SLICE, FakeNvml())
    assert gs.evaluate_nvml(SLICE, probe)["scope"] == "slice_verified"
    over = gs.evaluate_nvml(SLICE, {**probe, "memory_used_mib": 30000})
    assert over["scope"] == "scope_unverified" and over["reason"] == "memory_used_exceeds_total"
    assert gs.evaluate_nvml(SLICE, {**probe, "memory_total_mib": 0})["reason"] == "memory_used_exceeds_total"
    assert gs.evaluate_nvml(SLICE, {**probe, "memory_used_mib": None}) == {"scope": "unavailable", "reason": "nvml_memory_not_available"}
    assert gs.evaluate_nvml(SLICE, {**probe, "memory_used_mib": True})["scope"] == "unavailable"


def valid_record(**changes):
    return json.dumps({"allocated_uuid": SLICE, "parent_uuid": PARENT, "method": gs.CSV_METHOD, "sampler_csv_sha256": "0" * 64, "scope": "slice_verified", "reason": None, "parent_total_mib": 97887,
                       "torch_total_mib": None, "memory_total_mib": 24000, "returned_uuids": [SLICE], **changes})


@pytest.mark.parametrize("content", ["[]", "null", '"x"', "123", "true", "{}", '{"scope": ["a"]}', valid_record(parent_total_mib="97887"), valid_record(parent_total_mib=-5), valid_record(torch_total_mib=True),
                                     valid_record(allocated_uuid=[SLICE]), valid_record(returned_uuids="MIG"), valid_record(scope="great"), valid_record(method=None)])
def test_a_malformed_scope_record_is_unavailable_evidence_and_never_an_exception(tmp_path, content):
    csv_path = make_run(tmp_path, [row()])
    (csv_path.parent / gs.SCOPE_FILE).write_text(content, encoding="utf-8")
    report = gs.memory_report(csv_path)
    assert report["scope"] == "unavailable" and report["peak_mib"] is None and report["reason"].startswith("scope record malformed") and report["device_reported_peak_mib"] == 7900


def test_a_record_that_is_not_json_is_unavailable_too(tmp_path):
    csv_path = make_run(tmp_path, [row()])
    (csv_path.parent / gs.SCOPE_FILE).write_text("{not json", encoding="utf-8")
    report = gs.memory_report(csv_path)
    assert report["scope"] == "unavailable" and report["peak_mib"] is None and "cannot be read" in report["reason"]


@pytest.mark.parametrize("meta", ['[]', '{"gpu_uuid": ["MIG-aaaaaaaa-1111"]}', '{"gpu_uuid": 5}', '{"gpu_uuid": "MIG-aaaaaaaa-1111", "parent_gpu_uuid": ["GPU-bbbbbbbb-2222"]}'])
def test_malformed_run_meta_never_gives_a_number_and_never_raises(tmp_path, meta):
    csv_path = make_run(tmp_path, [row()])
    (csv_path.parent / "run-meta.json").write_text(meta, encoding="utf-8")
    report = gs.memory_report(csv_path)
    assert report["peak_mib"] is None and report["scope"] in ("unavailable", "scope_unverified")


def no_smi(*args, **kwargs):
    return types.SimpleNamespace(returncode=1, stdout="", stderr="")


def test_a_list_valued_identity_is_never_a_device_and_build_record_and_the_command_survive_it(tmp_path):
    assert gs.uuid_kind(["MIG-aaaaaaaa-1111"]) is None and gs.uuid_kind(None) is None and gs.uuid_kind(5) is None and gs.uuid_kind(" MIG-aaaaaaaa-1111 ") == "mig"
    csv_path = make_run(tmp_path, [row()])
    record = gs.build_record({"gpu_uuid": ["MIG-aaaaaaaa-1111"], "parent_gpu_uuid": {"x": 1}}, csv_path, run=no_smi, env={})
    assert record["scope"] == "scope_unverified" and record["reason"] == "identity_missing"
    meta_path = csv_path.parent / "meta-list.json"
    meta_path.write_text("[]", encoding="utf-8")
    assert gs.main(["--run-meta", str(meta_path), "--csv", str(csv_path), "--out", str(tmp_path / "out.json")]) == 0 and not (tmp_path / "out.json").exists()


def test_the_record_stores_the_torch_capacity_it_was_judged_with(tmp_path):
    csv_path = make_run(tmp_path, [row()])
    meta = json.loads((csv_path.parent / "run-meta.json").read_text(encoding="utf-8"))
    record = gs.build_record(meta, csv_path, run=no_smi, nvml=FakeNvml(), env={}, torch_total_mib=24000)
    assert record["torch_total_mib"] == 24000
