"""sft_report.py: the eight figures, what each compares (reference run vs E00, epoch choice) and the OPTIMISED/diagnostic titles. Synthetic saved numbers only."""
import json

import pytest

matplotlib = pytest.importorskip("matplotlib")
import matplotlib.figure  # noqa: E402

import sft_report as RP  # noqa: E402

LANGS = RP.LANGS


def wer_block(errors, words):
    return {"errors": errors, "reference_words": words, "rate": errors / words, "substitutions": errors // 2, "deletions": errors // 4, "insertions": errors - errors // 2 - errors // 4}


def metrics(scale):
    scope = {"ALL": {"wer": wer_block(int(100 * scale), 10000), "entity": wer_block(int(30 * scale), 1000)}}
    for lang in LANGS:
        scope[lang] = {"wer": wer_block(int(25 * scale), 2500), "entity": wer_block(int(8 * scale), 250)}
    return {"stages": {"raw": {"by_scope": scope, "entity_groups": {"value:T01": {"errors": int(5 * scale), "reference_words": 100}, "event:READBACK": {"errors": int(2 * scale), "reference_words": 50}}}}}


def make_run(root, name, objective, weight=2.0, scale=1.0, epochs=3):
    run = root / name
    (run / "eval").mkdir(parents=True)
    (run / "config.json").write_text(json.dumps({"config": {"run_id": name, "objective": objective, "entity_weight": weight}}))
    stats = lambda x: {"ce": 0.05 * x, "wce2": 0.055 * x, "ent_ce": 0.04 * x, "non_ent_ce": 0.05 * x, "tokens": 1000, "entity_tokens": 90}   # noqa: E731
    (run / "train_metrics.jsonl").write_text("".join(json.dumps({"update": 10 * (i + 1), **stats(1.0 / (i + 1))}) + "\n" for i in range(6)))
    (run / "epoch_summary.jsonl").write_text("".join(json.dumps({"epoch": e, "update": 20 * e, "validation": stats(1.0 / (e + 1))}) + "\n" for e in range(0, epochs + 1)))
    for e in range(1, epochs + 1):
        folder = run / "eval" / f"epoch-{e}"
        folder.mkdir()
        (folder / "metrics.json").write_text(json.dumps(metrics(scale * (1 + 0.1 * e))))
    return run


@pytest.fixture
def spy(monkeypatch):
    seen = {}

    def savefig(self, path, *a, **k):
        ax = self.axes[0]
        seen[path.name] = (ax.get_title(), [t.get_text() for t in ax.get_legend().get_texts()] if ax.get_legend() else [])
    monkeypatch.setattr(matplotlib.figure.Figure, "savefig", savefig)
    return seen


def test_eight_figures_with_the_new_numbering(tmp_path, spy):
    root = tmp_path / "runs"
    run = make_run(root, "E02-W4", "wce", 4.0)
    base = tmp_path / "E00"
    base.mkdir()
    (base / "metrics.json").write_text(json.dumps(metrics(3.0)))
    RP.main(["--run", str(run), "--baseline", str(base)])
    assert sorted(spy) == ["fig1_ce_vs_steps.png", "fig2_wce2_vs_steps.png", "fig3_generated_wer_entity_wer.png", "fig4_entity_vs_nonentity_ce.png", "fig5_wer_by_language.png",
                           "fig6_entity_wer_by_language.png", "fig7_sdi_breakdown.png", "fig8_entity_wer_by_category.png"]
    assert "diagnostic, not optimised" in spy["fig1_ce_vs_steps.png"][0] and "WCE@4" in spy["fig2_wce2_vs_steps.png"][0] and "OPTIMISED" in spy["fig2_wce2_vs_steps.png"][0]
    # no reference.json: the comparison is with E00, at epoch 2 (the provisional checkpoint), not the last epoch
    assert spy["fig5_wer_by_language.png"][1] == ["E00 (base)", "E02-W4 epoch-2"] and spy["fig6_entity_wer_by_language.png"][1] == ["E00 (base)", "E02-W4 epoch-2"]


def test_default_reference_from_reference_json_and_overrides(tmp_path, spy):
    root = tmp_path / "runs"
    make_run(root, "E01-LR25", "ce", scale=1.0)
    run = make_run(root, "E02-W4", "wce", 4.0, scale=1.1)
    (root / "reference.json").write_text(json.dumps({"run": "E01-LR25", "epoch": 2}))
    RP.main(["--run", str(run)])
    for fig in ("fig5_wer_by_language.png", "fig6_entity_wer_by_language.png"):
        assert spy[fig][1] == ["E01-LR25 epoch-2 (reference)", "E02-W4 epoch-2"]
    assert "E01-LR25 epoch-2 (reference) vs E02-W4 epoch-2" in spy["fig8_entity_wer_by_category.png"][0]
    spy.clear()
    base = tmp_path / "E00"
    base.mkdir()
    (base / "metrics.json").write_text(json.dumps(metrics(3.0)))
    RP.main(["--run", str(run), "--baseline", str(base), "--include-base", "--epoch", "3", "--reference-epoch", "2"])
    assert spy["fig5_wer_by_language.png"][1] == ["E00 (base)", "E01-LR25 epoch-2 (reference)", "E02-W4 epoch-3"]
    spy.clear()
    RP.main(["--run", str(run), "--baseline", str(base), "--no-reference"])
    assert spy["fig5_wer_by_language.png"][1] == ["E00 (base)", "E02-W4 epoch-2"]
    spy.clear()
    RP.main(["--run", str(root / "E01-LR25"), "--baseline", str(base)])                  # the reference run itself falls back to E00
    assert spy["fig5_wer_by_language.png"][1] == ["E00 (base)", "E01-LR25 epoch-2"] and "OPTIMISED" in spy["fig1_ce_vs_steps.png"][0]


def test_missing_epoch_is_refused(tmp_path, spy):
    root = tmp_path / "runs"
    run = make_run(root, "E02-W8", "wce", 8.0, epochs=2)
    with pytest.raises(SystemExit, match="no evaluation for epoch 3"):
        RP.main(["--run", str(run), "--epoch", "3"])
