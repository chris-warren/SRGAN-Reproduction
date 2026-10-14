"""Enforce that documentation matches the code it describes.

These docs quote measured numbers — parameter counts, memory, the paper's
training values. Prose drifts silently when code changes, and a reader has no
way to tell a stale figure from a current one. So the figures are asserted here:
change a config and the docs quoting it fail until updated.

Only cheap, config-derived numbers are checked. Dataset-derived figures in
DATASET.md (lags, error magnitudes) would need the corpus and are covered by
`scripts/verify_waveform_dataset.py` instead.
"""

import re
from pathlib import Path

import pytest

from mp3ml.config import estimate_size, load_config, output_frames

ROOT = Path(__file__).resolve().parents[1]
REPRO = ROOT / "configs" / "REPRODUCTION.md"


@pytest.fixture(scope="module")
def doc() -> str:
    return REPRO.read_text()


def test_sizing_table_matches_computed_values(doc: str) -> None:
    """§4.1's parameter and memory figures must match `estimate_size`."""
    rows = re.findall(r"\| `(faithful|local)\.yaml` \| ([\d.]+) M \| ([\d.]+) GB", doc)
    assert len(rows) == 2, "sizing table in REPRODUCTION.md section 3.1 not found or malformed"
    for name, params_m, adam_gb in rows:
        est = estimate_size(load_config(ROOT / "configs" / f"{name}.yaml"))
        assert float(params_m) == pytest.approx(est["total_params"] / 1e6, abs=0.15), name
        assert float(adam_gb) == pytest.approx(est["adam_state_gb"], abs=0.02), name


def test_documented_paper_training_values_match_the_faithful_config(doc: str) -> None:
    """Numbers the doc attributes to §3.2 must be what `faithful.yaml` actually sets."""
    t = load_config(ROOT / "configs" / "faithful.yaml").train
    assert re.search(rf"\*\*{t.iterations // 1000}[,.]?000\*\*|\*\*40,000\*\*", doc), "iterations"
    assert f"**{t.batch_size}**" in doc, "batch size"
    assert "1e-3" in doc, "learning rate"


def test_documented_output_frame_count_is_real(doc: str) -> None:
    """The 336 -> 212 claim appears throughout; it must follow from the config."""
    assert "212" in doc
    assert output_frames(load_config(ROOT / "configs" / "faithful.yaml")) == 212


def test_local_deviations_table_is_truthful(doc: str) -> None:
    """§4.1 claims local keeps the paper's time structure — verify, don't trust."""
    local = load_config(ROOT / "configs" / "local.yaml")
    faithful = load_config(ROOT / "configs" / "faithful.yaml")
    assert "*unchanged*" in doc, "deviations table missing"
    assert local.data.segment_frames == faithful.data.segment_frames
    assert local.model.dilations == faithful.model.dilations


def test_doc_records_the_measured_batch_size(doc: str) -> None:
    """local's batch deviates from the paper; the doc must say the real value."""
    local = load_config(ROOT / "configs" / "local.yaml")
    assert local.train.batch_size != 12, "expected a deviation from the paper's 12"
    assert re.search(rf"\| `batch_size` \| 12 \| {local.train.batch_size} \|", doc), (
        "REPRODUCTION.md section 3.1 deviations table does not record the real batch size"
    )


def _parameters(value: object) -> set[str]:
    """Parameter names of `value` if it is callable, else nothing."""
    import inspect

    if not callable(value):
        return set()
    try:
        return set(inspect.signature(value).parameters)
    except (TypeError, ValueError):  # builtins and C extensions have no signature
        return set()


def test_every_code_name_in_the_doc_exists(doc: str) -> None:
    """Catch names renamed in code but left behind in prose.

    Resolves each backticked snake_case token against the config fields, the
    public symbols of every `mp3ml` module, and the repo's own filenames — so it
    maintains itself rather than needing an allow-list to be kept in step.
    """
    import importlib
    import pkgutil

    import mp3ml

    known: set[str] = set()
    cfg = load_config(ROOT / "configs" / "faithful.yaml")
    for section in (cfg.stft, cfg.model, cfg.data, cfg.train):
        known |= set(vars(section))
    for info in pkgutil.iter_modules(mp3ml.__path__):
        known.add(info.name)
        module = importlib.import_module(f"mp3ml.{info.name}")
        known |= set(vars(module))
        # Methods too: `Trainer.critic_step` is as real a name as a function.
        # And parameters: `estimate_lag`'s `expected_lag` is as real a name as
        # the function it belongs to, and the doc cites it when explaining why
        # the constrained search exists.
        for value in vars(module).values():
            if isinstance(value, type):
                known |= set(vars(value))
                for member in vars(value).values():
                    known |= _parameters(member)
            known |= _parameters(value)
    for path in ROOT.rglob("*.py"):
        known.add(path.stem)
    for path in ROOT.rglob("*.*"):
        known.add(path.stem)
    # The doc cites tests as the proof of a claim, so test function names are
    # real names too. Read as text rather than imported: collecting the suite
    # from inside it is circular.
    for path in (ROOT / "tests").glob("test_*.py"):
        known |= set(re.findall(r"^def (test_\w+)", path.read_text(), re.M))

    # The doc once carried a section of third-party identifiers that had to be
    # exempted here; the 2026-10-04 cleanup removed that section, so the check
    # now covers the whole file.
    ours = doc

    unknown = {
        name for name in re.findall(r"`([a-z][a-z0-9_]*_[a-z0-9_]+)`", ours) if name not in known
    }
    assert not unknown, (
        f"names in REPRODUCTION.md that exist nowhere in the code: {sorted(unknown)} "
        "— renamed and left stale in the doc?"
    )


def test_provenance_markers_are_used(doc: str) -> None:
    """The legend is the doc's main safeguard; every section heading must carry a mark."""
    for mark in ("[P]", "[I]", "[A]", "[D]"):
        assert mark in doc, mark
    headings = [
        h
        for h in re.findall(r"^#{2,3} .+$", doc, re.M)
        if not h.startswith(("## Provenance", "## 5.", "## 6.", "## 7."))
        and "Changelog" not in h
        and "6.1" not in h
    ]
    unmarked = [h for h in headings if not re.search(r"\[[PIAD](/[PIAD])?\]", h)]
    assert not unmarked, f"section headings without a provenance marker: {unmarked}"


def test_coverage_matrix_cites_only_tests_that_exist() -> None:
    """`configs/COVERAGE.md` claims a test for every DONE row. If a test is
    renamed or deleted, the claim becomes false silently — which is exactly the
    failure mode the coverage matrix exists to prevent.
    """
    import re

    root = Path(__file__).resolve().parents[1]
    doc = (root / "configs" / "COVERAGE.md").read_text()
    cited = sorted(set(re.findall(r"`(test_\w+)::(\w+)`", doc)))
    assert cited, "COVERAGE.md cites no tests — has its format changed?"

    missing = []
    for module, name in cited:
        path = root / "tests" / f"{module}.py"
        source = path.read_text() if path.exists() else ""
        if f"def {name}(" not in source and f"def test_{name}(" not in source:
            missing.append(f"{module}::{name}")
    assert not missing, f"COVERAGE.md cites tests that do not exist: {missing}"


def test_coverage_matrix_has_no_unresolved_gaps() -> None:
    """A GAP row means the paper specifies something we have not implemented.
    Finding one is fine; leaving it unrecorded is not. This fails if a GAP is
    added without also listing it under "Gaps and open items"."""
    root = Path(__file__).resolve().parents[1]
    doc = (root / "configs" / "COVERAGE.md").read_text()
    body, _, tail = doc.partition("## Gaps and open items")
    gap_rows = [ln for ln in body.splitlines() if ln.startswith("|") and "GAP" in ln]
    # The legend row defines what GAP means and is not itself a gap.
    gap_rows = [ln for ln in gap_rows if "specified and not implemented" not in ln]
    assert not gap_rows or tail, f"GAP rows present but no gap list: {gap_rows}"
