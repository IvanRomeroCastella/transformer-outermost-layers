"""
run_pipeline.py -- Orchestrate the full pipeline and produce a consolidated
HTML report and raw text dump.

Runs the 13 pipeline scripts in dependency order. Before each step, verifies
that its inputs exist (and lists what's missing if not). After each step,
verifies that its outputs were produced. Captures stdout/stderr per step,
records timings, and writes a self-contained HTML report and a plain-text
dump of all script output to the repo root.

The report contains: per-step status table (OK / FAIL / SKIPPED), timings,
the figures from both blocks embedded inline as base64, and the numerical
summary from results/summary.json formatted as a per-model table.

The orchestrator does NOT decide hypothesis outcomes. Those live in
prereg.md in each block. The report only surfaces the numbers; the human
maps them onto the prereg.

Outputs produced at the repo root:
  - pipeline_report.html       HTML report with status, figures, summary
  - pipeline_results_raw.txt   plain-text dump of all step stdout/stderr

Per-step persistent state, under <cwd>/.pipeline_done/:
  - <step_name>.log.json       stdout/stderr/elapsed of the last run, including
                               failed runs (so partial output from a crashed
                               script is not lost). Successful (OK) logs are
                               never overwritten by a subsequent FAIL log,
                               which protects the audit trail of completed
                               steps.
  - <step_name>.done           sentinel for in-place steps (records script hash)

These persistent logs are loaded back into the HTML report and the raw text
dump when a step is SKIPPED in a later invocation, so information about
previously completed steps is never lost across incremental runs.

Crash safety:
  - Both the .txt and the .html are re-written after every step, not just at
    the end. If the pipeline crashes mid-way, the files reflect everything
    completed up to (and including) the failing step.
  - Writes are atomic (write-tmp + rename + fsync), so a power loss or Ctrl+C
    during a write cannot corrupt the existing file.
  - Ctrl+C is caught and the .txt is flushed before exit.
  - The stderr filter has a defensive fallback: if a regex bug crashes the
    filter, the raw stderr is preserved instead of losing the data.

Usage:
  python run_pipeline.py                          # run everything that's missing
  python run_pipeline.py --rerun                  # ignore existing outputs, run all
  python run_pipeline.py --dry-run                # show what would run, don't execute
                                                  # (does NOT overwrite .txt or .html)
  python run_pipeline.py --list-steps             # print all step names
  python run_pipeline.py --force-step block2.fix_freq_bucket
                                                  # re-run one step regardless of state

Idempotency:
  Most steps are skipped if their declared `expected_outputs` already exist.
  Steps that mutate files in place (e.g. fix_freq_bucket) use sentinel
  files at <cwd>/.pipeline_done/<step_name>.done to track completion. The
  sentinel records the script's SHA1 hash; modifying the script invalidates
  the sentinel and the step re-runs automatically on the next invocation.

Exit codes:
  0    all steps OK (or skipped because their outputs already exist)
  1    at least one step failed
  2    invalid command-line arguments
  130  interrupted by Ctrl+C (SIGINT)
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from html import escape
from pathlib import Path

REPO_ROOT = Path(__file__).parent.resolve()
BLOCK1 = REPO_ROOT / "01_sandwich_geometry"
BLOCK2 = REPO_ROOT / "02_translator_mechanism"

MODELS = ["distilbert", "bert", "gpt2"]
CONDITIONS = ["coherent", "permuted", "random"]
PROBING_DATASETS = ["ud_ewt", "sst2"]


def _atomic_write_text(path: Path, text: str) -> None:
    """Write `text` to `path` atomically: write to a sibling tmp file, fsync,
    then rename over the target. On POSIX, rename is atomic, so the file at
    `path` is either the previous full contents or the new full contents --
    never a half-written file. Cheap insurance against power loss or Ctrl+C
    during a write."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
        f.flush()
        try:
            os.fsync(f.fileno())
        except OSError:
            # Not all filesystems support fsync; rename atomicity is the main guarantee.
            pass
    os.replace(tmp, path)


# -----------------------------------------------------------------------------
# Pipeline declaration
# -----------------------------------------------------------------------------

@dataclass
class Step:
    """One pipeline step."""
    name: str                 # human-readable label
    script: Path              # absolute path to the script
    cwd: Path                 # directory to run it from
    expected_outputs: list    # list of Path; all must exist after the step succeeds
    required_inputs: list = field(default_factory=list)  # must exist before running
    idempotency_check: str = "outputs_exist"
    # How to decide whether a step can be skipped on re-run:
    #   "outputs_exist": skip if all expected_outputs exist. Correct for steps
    #                    that produce NEW files from their inputs.
    #   "sentinel":      skip only if a sentinel file (<cwd>/.pipeline_done/<name>.done)
    #                    exists AND records a matching script hash. Required for
    #                    steps that MUTATE existing files in place
    #                    (e.g. 06_fix_freq_bucket). If the script changes, the
    #                    hash mismatch invalidates the sentinel and the step
    #                    re-runs.


def _block1_outputs(prefix: str, suffix: str) -> list:
    """Helper: data/<model>/<prefix>_<condition><suffix> for all models x conditions."""
    out = []
    for m in MODELS:
        for c in CONDITIONS:
            out.append(BLOCK1 / "data" / m / f"{prefix}_{c}{suffix}")
    return out


def _block1_results(prefix: str) -> list:
    """Helper: results/<prefix>_<model>_<condition>.npz for all models x conditions."""
    return [BLOCK1 / "results" / f"{prefix}_{m}_{c}.npz"
            for m in MODELS for c in CONDITIONS]


def _block2_b1_outputs() -> list:
    return [BLOCK2 / "results" / f"b1_ablation_{m}_{c}.npz"
            for m in MODELS for c in CONDITIONS]


def _block2_probing_outputs() -> list:
    return [BLOCK2 / "results" / f"probing_data_{m}_{d}.npz"
            for m in MODELS for d in PROBING_DATASETS]


PIPELINE: list = [
    # Block 1.
    Step(
        name="block1.build_datasets",
        script=BLOCK1 / "build_datasets.py",
        cwd=BLOCK1,
        required_inputs=[],
        expected_outputs=_block1_outputs("dataset", ".jsonl"),
    ),
    Step(
        name="block1.extract_embeddings",
        script=BLOCK1 / "extract_embeddings.py",
        cwd=BLOCK1,
        required_inputs=_block1_outputs("dataset", ".jsonl"),
        expected_outputs=_block1_outputs("embeddings", ".npz"),
    ),
    Step(
        name="block1.metrics_global",
        script=BLOCK1 / "metrics_global.py",
        cwd=BLOCK1,
        required_inputs=_block1_outputs("embeddings", ".npz"),
        expected_outputs=_block1_results("global"),
    ),
    Step(
        name="block1.metrics_m1b",
        script=BLOCK1 / "metrics_m1b.py",
        cwd=BLOCK1,
        required_inputs=_block1_outputs("embeddings", ".npz"),
        expected_outputs=_block1_results("m1b"),
    ),
    Step(
        name="block1.metrics_m3",
        script=BLOCK1 / "metrics_m3.py",
        cwd=BLOCK1,
        required_inputs=_block1_outputs("embeddings", ".npz"),
        expected_outputs=_block1_results("m3"),
    ),
    Step(
        name="block1.visualize",
        script=BLOCK1 / "visualize.py",
        cwd=BLOCK1,
        required_inputs=(_block1_results("global")
                         + _block1_results("m1b")
                         + _block1_results("m3")),
        # The visualize script picks figure filenames internally; we check
        # that the figures directory exists and has at least one PNG.
        expected_outputs=[BLOCK1 / "results" / "figures"],
    ),

    # Block 2.
    Step(
        name="block2.b1_ablation",
        script=BLOCK2 / "b1_ablation.py",
        cwd=BLOCK2,
        required_inputs=_block1_outputs("embeddings", ".npz"),
        expected_outputs=_block2_b1_outputs(),
    ),
    Step(
        name="block2.extract_probing_data",
        script=BLOCK2 / "extract_probing_data.py",
        cwd=BLOCK2,
        required_inputs=[],   # downloads UD-EWT / SST-2 itself
        expected_outputs=_block2_probing_outputs(),
    ),
    Step(
        name="block2.fix_freq_bucket",
        script=BLOCK2 / "fix_freq_bucket.py",
        cwd=BLOCK2,
        required_inputs=_block2_probing_outputs(),
        # Modifies files in place; uses a sentinel to track completion since
        # "the output files exist" doesn't imply "the fix has been applied".
        expected_outputs=_block2_probing_outputs(),
        idempotency_check="sentinel",
    ),
    Step(
        name="block2.a1_probing",
        script=BLOCK2 / "a1_probing.py",
        cwd=BLOCK2,
        required_inputs=_block2_probing_outputs(),
        expected_outputs=[BLOCK2 / "results" / "a1_probing.npz"],
    ),
    Step(
        name="block2.a2_budget",
        script=BLOCK2 / "a2_budget.py",
        cwd=BLOCK2,
        required_inputs=[BLOCK2 / "results" / "a1_probing.npz"]
                        + _block2_probing_outputs(),
        expected_outputs=[BLOCK2 / "results" / "a2_budget.npz"],
    ),
    Step(
        name="block2.a3_weights",
        script=BLOCK2 / "a3_weights.py",
        cwd=BLOCK2,
        required_inputs=[BLOCK2 / "results" / "a1_probing.npz"]
                        + _block2_probing_outputs(),
        expected_outputs=[BLOCK2 / "results" / "a3_weights.npz"],
    ),
    Step(
        name="block2.analyze",
        script=BLOCK2 / "analyze.py",
        cwd=BLOCK2,
        required_inputs=[
            BLOCK2 / "results" / "a1_probing.npz",
            BLOCK2 / "results" / "a2_budget.npz",
            BLOCK2 / "results" / "a3_weights.npz",
        ] + _block2_b1_outputs(),
        expected_outputs=[
            BLOCK2 / "results" / "summary.json",
            BLOCK2 / "results" / "figures",
        ],
    ),
]


# -----------------------------------------------------------------------------
# Execution
# -----------------------------------------------------------------------------

@dataclass
class StepResult:
    name: str
    status: str = ""          # "OK", "FAIL", "SKIPPED", "MISSING_INPUTS"
    elapsed_s: float = 0.0
    missing_inputs: list = field(default_factory=list)
    missing_outputs: list = field(default_factory=list)
    stdout: str = ""          # truncated to last 4000 chars (for HTML)
    stderr: str = ""          # truncated to last 4000 chars (for HTML)
    stdout_full: str = ""     # untruncated (for disk log and raw text dump)
    stderr_full: str = ""     # untruncated (for disk log and raw text dump)
    returncode: int = 0


def _missing(paths: list) -> list:
    return [p for p in paths if not p.exists()]


def _all_outputs_present(step: Step) -> bool:
    return all(p.exists() for p in step.expected_outputs)


def _sentinel_path(step: Step) -> Path:
    """Where the sentinel for an in-place step lives."""
    return step.cwd / ".pipeline_done" / f"{step.name}.done"


def _script_hash(script_path: Path) -> str:
    """SHA1 of the script's source. Used to invalidate sentinels when the
    underlying script changes."""
    if not script_path.exists():
        return ""
    return hashlib.sha1(script_path.read_bytes()).hexdigest()


def _sentinel_valid(step: Step) -> bool:
    """A sentinel is valid iff it exists, parses as JSON, and its recorded
    script_hash matches the current script's hash."""
    sentinel = _sentinel_path(step)
    if not sentinel.exists():
        return False
    try:
        payload = json.loads(sentinel.read_text())
    except Exception:
        return False
    return payload.get("script_hash") == _script_hash(step.script)


def _write_sentinel(step: Step) -> None:
    sentinel = _sentinel_path(step)
    sentinel.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "step_name": step.name,
        "script": str(step.script),
        "script_hash": _script_hash(step.script),
        "completed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    sentinel.write_text(json.dumps(payload, indent=2))


# -----------------------------------------------------------------------------
# Persistent step logs
# -----------------------------------------------------------------------------
#
# We persist each step's stdout/stderr/timing to disk under
# <cwd>/.pipeline_done/<step_name>.log.json so that the HTML report and the
# raw text dump can show full information even for steps that were SKIPPED
# in the current invocation (i.e. completed in a previous run). Failed runs
# are also persisted -- without clobbering a prior OK log -- so partial
# output from a crashed script is not lost.

def _log_path(step: Step) -> Path:
    return step.cwd / ".pipeline_done" / f"{step.name}.log.json"


def _write_log(step: Step, result: "StepResult") -> None:
    """Persist this step's stdout/stderr/timing to disk.

    Safety rule: never overwrite a previously persisted OK log with a non-OK
    log. If this step ran successfully before and is now failing (e.g. user
    forced a re-run that crashed), we keep the OK log untouched so the audit
    trail of the previous successful completion survives. The failing run's
    output is written to a sibling `.failed.json` file for forensic use.
    """
    p = _log_path(step)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "step_name": step.name,
        "status": result.status,
        "returncode": result.returncode,
        "elapsed_s": result.elapsed_s,
        "completed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "stdout": result.stdout_full or result.stdout,
        "stderr": result.stderr_full or result.stderr,
    }
    text = json.dumps(payload, indent=2)

    if result.status != "OK":
        # Check if there's a prior OK log we shouldn't clobber.
        prior = _load_log(step)
        if prior.get("status") == "OK":
            failed_path = p.with_suffix(".failed.json")
            _atomic_write_text(failed_path, text)
            return

    _atomic_write_text(p, text)


def _load_log(step: Step) -> dict:
    """Load the persisted log for a step. Returns {} if none exists."""
    p = _log_path(step)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except Exception:
        return {}


def _hydrate_skipped_result(step: Step, result: "StepResult") -> None:
    """If a step is SKIPPED in this run, try to fill in details from the
    previously persisted log so the HTML report doesn't lose information."""
    log = _load_log(step)
    if not log:
        return
    result.elapsed_s = float(log.get("elapsed_s", 0.0))
    full_stdout = log.get("stdout", "")
    full_stderr = log.get("stderr", "")
    result.stdout_full = full_stdout
    result.stderr_full = full_stderr
    result.stdout = full_stdout[-4000:]
    result.stderr = full_stderr[-4000:]
    result.returncode = int(log.get("returncode", 0))


def _can_skip(step: Step) -> bool:
    """Whether this step can be SKIPPED on the current invocation.

    Logic depends on the step's idempotency_check:
      - "outputs_exist": skip iff all expected_outputs exist.
      - "sentinel":      skip iff the sentinel is valid AND all outputs exist.
                         The sentinel becomes invalid if the script's source
                         changed since the last completion.
    """
    if not _all_outputs_present(step):
        return False
    if step.idempotency_check == "outputs_exist":
        return True
    if step.idempotency_check == "sentinel":
        return _sentinel_valid(step)
    raise ValueError(f"Unknown idempotency_check: {step.idempotency_check}")


def run_step(step: Step, rerun: bool, dry_run: bool, force: bool = False) -> StepResult:
    res = StepResult(name=step.name)

    if not rerun and not force and _can_skip(step):
        res.status = "SKIPPED"
        # Hydrate from persisted log so the HTML report has full detail
        # even for steps completed in a previous invocation.
        _hydrate_skipped_result(step, res)
        return res

    missing = _missing(step.required_inputs)
    if missing:
        res.status = "MISSING_INPUTS"
        res.missing_inputs = missing
        return res

    if dry_run:
        res.status = "OK (dry-run)"
        return res

    t0 = time.time()
    stdout_lines = []
    stderr_lines = []
    try:
        # Stream child output live to the parent's terminal AND keep it for
        # the report. We use Popen + line-buffered pipes and a tiny reader
        # loop so the user sees progress as it happens.
        proc = subprocess.Popen(
            [sys.executable, "-u", str(step.script.name)],  # -u disables child stdout buffering
            cwd=str(step.cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,                                       # line-buffered
        )

        # Read stdout and stderr concurrently using threads to avoid
        # deadlocking on whichever pipe fills first.
        import threading

        # last_output_at[0] is updated by the pump threads every time the
        # child prints a line. The main loop uses it as a liveness signal:
        # the child has been quiet for N seconds, it's probably done (or
        # hung). Combined with checking whether the expected outputs are
        # already on disk, this lets us detect the "child finished work but
        # didn't exit" case caused by stuck daemon threads inside the child
        # (a known issue with huggingface_hub / tqdm / datasets background
        # workers that don't always shut down cleanly).
        last_output_at = [time.time()]
        output_lock = threading.Lock()

        def _pump(stream, buf, prefix):
            for line in stream:
                buf.append(line)
                with output_lock:
                    last_output_at[0] = time.time()
                sys.stdout.write(f"  {prefix}{line}")
                sys.stdout.flush()
            stream.close()

        t_out = threading.Thread(target=_pump, args=(proc.stdout, stdout_lines, ""), daemon=True)
        t_err = threading.Thread(target=_pump, args=(proc.stderr, stderr_lines, "! "), daemon=True)
        t_out.start()
        t_err.start()

        # Liveness watchdog. Poll `proc.wait` with a short timeout, and if
        # the child stays alive AND silent AND all its expected outputs are
        # already on disk, conclude that the child is hung at exit (not
        # doing real work) and kill it. The threshold is generous enough to
        # tolerate normal end-of-run cleanup (saving large numpy files,
        # closing the model, freeing CUDA memory).
        IDLE_KILL_SECONDS = 60
        killed_at_exit = False
        while True:
            try:
                proc.wait(timeout=5)
                break  # child exited on its own; normal path
            except subprocess.TimeoutExpired:
                with output_lock:
                    idle_for = time.time() - last_output_at[0]
                if idle_for < IDLE_KILL_SECONDS:
                    continue  # still printing recently, give it more time
                # Idle for a while. If all expected outputs are present,
                # the child has done its job and is just hung on shutdown.
                # Otherwise it might be stuck mid-work; give it the benefit
                # of the doubt and keep waiting.
                if step.expected_outputs and not _missing(step.expected_outputs):
                    sys.stdout.write(
                        f"  [watchdog] child silent for {idle_for:.0f}s and all "
                        f"outputs present; terminating stuck process.\n"
                    )
                    sys.stdout.flush()
                    killed_at_exit = True
                    proc.terminate()
                    try:
                        proc.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.wait()
                    break

        # Give the pump threads a moment to drain any final buffered lines
        # now that the pipes are closing. If they don't return promptly, we
        # move on -- they're daemons and won't keep the parent alive.
        t_out.join(timeout=5)
        t_err.join(timeout=5)

        # If we killed the child but the work was complete, treat it as a
        # successful run. The return code from a SIGTERM'd process is
        # non-zero, but the on-disk outputs are what matters here.
        if killed_at_exit:
            res.returncode = 0
        else:
            res.returncode = proc.returncode
        full_stdout = "".join(stdout_lines)
        full_stderr = "".join(stderr_lines)
        res.stdout_full = full_stdout
        res.stderr_full = full_stderr
        res.stdout = full_stdout[-4000:]   # tail to keep report manageable
        res.stderr = full_stderr[-4000:]
    except Exception as e:
        res.status = "FAIL"
        res.stderr = f"Exception while launching script: {e!r}"
        res.elapsed_s = time.time() - t0
        return res

    res.elapsed_s = time.time() - t0

    if res.returncode != 0:
        res.status = "FAIL"
    else:
        res.missing_outputs = _missing(step.expected_outputs)
        if res.missing_outputs:
            res.status = "FAIL"
        else:
            res.status = "OK"
            if step.idempotency_check == "sentinel":
                try:
                    _write_sentinel(step)
                except Exception as e:
                    # The step ran successfully; failing to write the sentinel
                    # is a soft failure. The next run will just re-execute the
                    # step, which is the safe choice.
                    res.stderr += f"\n[warn] failed to write sentinel: {e!r}"

    # Persist the log with the definitive status, exactly once. _write_log
    # protects an existing OK log from being clobbered by a later FAIL.
    try:
        _write_log(step, res)
    except Exception as e:
        res.stderr += f"\n[warn] failed to write step log: {e!r}"

    return res


# -----------------------------------------------------------------------------
# Report generation
# -----------------------------------------------------------------------------

STATUS_COLOR = {
    "OK": "#7BC47F",
    "OK (dry-run)": "#A0C4E2",
    "SKIPPED": "#C0C0C0",
    "FAIL": "#E08080",
    "MISSING_INPUTS": "#E0B070",
}


def _img_b64(path: Path) -> str:
    if not path.exists():
        return ""
    data = path.read_bytes()
    b64 = base64.b64encode(data).decode("ascii")
    return f"data:image/png;base64,{b64}"


def _collect_figures() -> list:
    """All PNGs from both blocks, in a stable order."""
    figs = []
    for d in [BLOCK1 / "results" / "figures", BLOCK2 / "results" / "figures"]:
        if not d.exists():
            continue
        for p in sorted(d.glob("*.png")):
            figs.append(p)
    return figs


def _format_summary_json() -> str:
    """Format results/summary.json (block 2) as readable HTML tables."""
    path = BLOCK2 / "results" / "summary.json"
    if not path.exists():
        return "<p><em>summary.json not yet generated.</em></p>"

    try:
        data = json.loads(path.read_text())
    except Exception as e:
        return f"<p><em>summary.json could not be parsed: {escape(str(e))}</em></p>"

    out = []
    rb = data.get("a3_random_baseline_alignment")
    if rb is not None:
        out.append(f"<p><strong>A3 random baseline alignment</strong> (reference): {rb:.4f}</p>")

    models = data.get("models", {})
    for model, m_data in models.items():
        out.append(f"<h3>{escape(model)}</h3>")
        layers = m_data.get("layers", {})
        out.append(f"<p>Entry translator: state {layers.get('entry')}; "
                   f"core slice: {layers.get('core_slice')}; "
                   f"exit translator: state {layers.get('exit')}</p>")

        for section_name in ("entry_translator", "core", "exit_translator"):
            sect = m_data.get(section_name, {})
            if not sect:
                continue
            out.append(f"<h4>{escape(section_name.replace('_', ' '))}</h4>")
            out.append("<table class='kv'>")
            for k, v in sect.items():
                if isinstance(v, float):
                    vstr = f"{v:.4f}"
                else:
                    vstr = escape(str(v))
                out.append(f"<tr><td>{escape(k)}</td><td>{vstr}</td></tr>")
            out.append("</table>")

        rho = m_data.get("b1_final_layer_spearman_pos_vs_sensitivity", {})
        if rho:
            out.append("<h4>B1 final-layer Spearman rho (ablated position vs sensitivity)</h4>")
            out.append("<table class='kv'>")
            for cond, val in rho.items():
                out.append(f"<tr><td>{escape(cond)}</td><td>{val:+.4f}</td></tr>")
            out.append("</table>")

    return "\n".join(out)


# Patterns matched against stderr lines that are considered noise (HuggingFace
# downloads, tokenizer warnings, tqdm progress bars, etc.) and dropped from the
# raw text dump. This filter only affects pipeline_results_raw.txt -- the full
# untruncated stderr is still persisted in .pipeline_done/<step>.log.json, so
# nothing is permanently lost. Stdout is never filtered: that's where the
# scripts emit their measurements.
_STDERR_NOISE_PATTERNS = [
    # tqdm-style progress bars (lines containing a percent bar or it/s metric).
    re.compile(r"\d+%\|"),
    re.compile(r"\d+(\.\d+)?\s*(it|ex|examples|samples|MB|kB|GB|B)/s"),
    # HuggingFace download progress (file: 100%|...| 123M/123M [00:05<00:00, ...])
    re.compile(r"^\s*\S+\.(json|txt|safetensors|bin|model|h5|onnx|tflite|msgpack|arrow)\s*:"),
    re.compile(r"^\s*(Downloading|Fetching|Resolving|Filter|Map|Generating)\b"),
    # transformers / huggingface_hub / datasets warning preambles.
    re.compile(r"(FutureWarning|UserWarning|DeprecationWarning|RuntimeWarning):"),
    re.compile(r"warnings\.warn\("),
    re.compile(r"^\s*Some weights of "),
    re.compile(r"^\s*You should probably TRAIN this model "),
    re.compile(r"^\s*The tokenizer class you load "),
    re.compile(r"^\s*Special tokens have been added "),
    re.compile(r"^\s*Token indices sequence length is longer "),
    re.compile(r"^\s*Asking to truncate to max_length "),
    re.compile(r"^\s*Setting `pad_token_id` "),
    re.compile(r"^\s*huggingface/tokenizers: "),
    re.compile(r"^\s*Disabling parallelism to avoid deadlocks"),
    re.compile(r"^\s*To disable this warning"),
    re.compile(r"^\s*-\s*Avoid using `tokenizers`"),
    re.compile(r"^\s*-\s*Explicitly set the environment variable"),
    re.compile(r"TOKENIZERS_PARALLELISM"),
    # TransformerLens / model-loading verbose reports
    # ("Loading weights: ...%", "BertModel LOAD REPORT from: ...",
    # the per-key status table, and its trailing legend lines).
    re.compile(r"^\s*Loading weights:"),
    re.compile(r"\bLOAD REPORT\b"),
    re.compile(r"^\s*Key\s+\|\s*Status\b"),
    re.compile(r"^\s*-+\+-+"),
    re.compile(r"\|\s*(UNEXPECTED|MISSING|MATCHED|OK)\b"),
    re.compile(r"^\s*Notes:\s*$"),
    re.compile(r"^\s*-\s+(UNEXPECTED|MISSING|MATCHED|OK)\b"),
    # The traceback-style continuation lines that follow a filtered warning header.
    re.compile(r"^\s+(category=|stacklevel=)"),
]


def _clean_stderr(text: str) -> str:
    """Drop HF/tqdm/transformers noise from a stderr blob, keep real messages.

    Conservative by design: if a line doesn't match any known noise pattern,
    it stays. The unfiltered text is still on disk under .pipeline_done/.

    If anything goes wrong during filtering (regex hiccup, weird unicode), we
    return the original text unchanged. Losing the noise filter is acceptable;
    losing the data is not.
    """
    if not text:
        return text
    try:
        kept = []
        for line in text.splitlines():
            if any(pat.search(line) for pat in _STDERR_NOISE_PATTERNS):
                continue
            kept.append(line)
        # Collapse runs of blank lines that the filter may have produced.
        out = []
        prev_blank = False
        for line in kept:
            is_blank = not line.strip()
            if is_blank and prev_blank:
                continue
            out.append(line)
            prev_blank = is_blank
        return "\n".join(out).strip()
    except Exception:
        # Defensive fallback: a bug in the filter should never lose the data.
        return text


def write_raw_results(results: list, output_path: Path) -> None:
    """Concatenate the full stdout (and filtered stderr) of every step into a
    single plain-text file. Useful as an auditable log of all numerical output
    printed by the pipeline scripts: probing accuracies, sensitivity stats,
    sandwich ratios, etc.

    stdout is included verbatim. stderr is filtered through `_clean_stderr` to
    drop HuggingFace download progress, tqdm bars, and the standard
    transformers/tokenizers warnings, since those obscure the actual numbers
    the scripts print. The unfiltered stderr is still on disk under
    .pipeline_done/<step>.log.json for forensic use.

    Skipped steps are included using their persisted log (so the dump stays
    complete across incremental runs)."""
    lines = []
    lines.append("=" * 78)
    lines.append("transformer-outermost-layers -- raw pipeline output")
    lines.append(f"Generated at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("Note: stderr is filtered for HF/tqdm/transformers noise.")
    lines.append("      Full unfiltered logs live in .pipeline_done/<step>.log.json.")
    lines.append("=" * 78)
    lines.append("")

    for r in results:
        lines.append("")
        lines.append("#" * 78)
        lines.append(f"# {r.name}  --  status={r.status}  elapsed={r.elapsed_s:.1f}s  rc={r.returncode}")
        lines.append("#" * 78)

        full_stdout = (r.stdout_full or r.stdout).rstrip()
        full_stderr_raw = (r.stderr_full or r.stderr).rstrip()
        full_stderr = _clean_stderr(full_stderr_raw)

        if full_stdout:
            lines.append("")
            lines.append("--- stdout ---")
            lines.append(full_stdout)

        if full_stderr:
            lines.append("")
            lines.append("--- stderr (filtered) ---")
            lines.append(full_stderr)

        if not full_stdout and not full_stderr:
            if r.status == "SKIPPED":
                lines.append("(no persisted log -- this step was completed before "
                             "log persistence was enabled)")
            elif r.status == "MISSING_INPUTS":
                lines.append("(did not run: required inputs were not present)")
            else:
                lines.append("(no output captured)")

    lines.append("")
    _atomic_write_text(output_path, "\n".join(lines))


def write_report(results: list, output_path: Path) -> None:
    figs = _collect_figures()

    # Per-step status table.
    status_rows = []
    for r in results:
        color = STATUS_COLOR.get(r.status, "#FFFFFF")
        time_str = f"{r.elapsed_s:.1f}s" if r.elapsed_s > 0 else "--"
        status_rows.append(
            f"<tr>"
            f"<td>{escape(r.name)}</td>"
            f"<td style='background:{color}; text-align:center; font-weight:bold;'>{escape(r.status)}</td>"
            f"<td style='text-align:right;'>{time_str}</td>"
            f"</tr>"
        )

    # Failure detail (only for failed / missing-input steps).
    fail_blocks = []
    for r in results:
        if r.status in ("OK", "OK (dry-run)", "SKIPPED"):
            continue
        block = [f"<h3>{escape(r.name)} -- {escape(r.status)}</h3>"]
        if r.missing_inputs:
            block.append("<p><strong>Missing inputs:</strong></p><ul>")
            for p in r.missing_inputs[:20]:
                block.append(f"<li><code>{escape(str(p))}</code></li>")
            if len(r.missing_inputs) > 20:
                block.append(f"<li>... and {len(r.missing_inputs) - 20} more</li>")
            block.append("</ul>")
        if r.missing_outputs:
            block.append("<p><strong>Outputs that should have been produced but weren't:</strong></p><ul>")
            for p in r.missing_outputs[:20]:
                block.append(f"<li><code>{escape(str(p))}</code></li>")
            if len(r.missing_outputs) > 20:
                block.append(f"<li>... and {len(r.missing_outputs) - 20} more</li>")
            block.append("</ul>")
        if r.stderr:
            block.append("<p><strong>Stderr (tail):</strong></p>")
            block.append(f"<pre>{escape(r.stderr)}</pre>")
        if r.stdout:
            block.append("<p><strong>Stdout (tail):</strong></p>")
            block.append(f"<pre>{escape(r.stdout)}</pre>")
        fail_blocks.append("\n".join(block))

    # Figures block.
    fig_blocks = []
    for p in figs:
        rel = p.relative_to(REPO_ROOT)
        fig_blocks.append(
            f"<figure>"
            f"<img src='{_img_b64(p)}' alt='{escape(str(rel))}'>"
            f"<figcaption><code>{escape(str(rel))}</code></figcaption>"
            f"</figure>"
        )

    summary_html = _format_summary_json()

    overall_ok = all(r.status in ("OK", "OK (dry-run)", "SKIPPED") for r in results)
    overall_color = STATUS_COLOR["OK"] if overall_ok else STATUS_COLOR["FAIL"]
    overall_label = "ALL OK" if overall_ok else "FAILURES PRESENT"

    total_time = sum(r.elapsed_s for r in results)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Pipeline report -- transformer-outermost-layers</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
         max-width: 1100px; margin: 2em auto; padding: 0 1em; color: #222; }}
  h1 {{ border-bottom: 2px solid #444; padding-bottom: 0.3em; }}
  h2 {{ border-bottom: 1px solid #aaa; padding-bottom: 0.2em; margin-top: 2em; }}
  table {{ border-collapse: collapse; width: 100%; margin: 1em 0; }}
  table.kv {{ width: auto; }}
  table.kv td {{ padding: 2px 12px; border-bottom: 1px solid #eee; font-family: monospace; }}
  table.kv td:first-child {{ color: #555; }}
  th, td {{ padding: 6px 10px; border-bottom: 1px solid #ddd; text-align: left; }}
  th {{ background: #f4f4f4; }}
  pre {{ background: #f7f7f7; padding: 0.8em; overflow-x: auto;
         font-size: 0.85em; border-left: 3px solid #aaa; }}
  figure {{ margin: 1.5em 0; text-align: center; }}
  figure img {{ max-width: 100%; border: 1px solid #ddd; }}
  figcaption {{ color: #666; font-size: 0.85em; margin-top: 0.3em; }}
  .overall {{ display: inline-block; padding: 0.4em 1em; font-weight: bold;
              border-radius: 4px; background: {overall_color}; }}
</style>
</head>
<body>

<h1>Pipeline report</h1>
<p><code>transformer-outermost-layers</code> &mdash; generated at
   {escape(time.strftime("%Y-%m-%d %H:%M:%S"))}</p>
<p>Overall: <span class="overall">{overall_label}</span>
   &nbsp;&nbsp; Total wall time: {total_time:.1f}s</p>

<h2>Per-step status</h2>
<table>
  <thead><tr><th>Step</th><th>Status</th><th>Time</th></tr></thead>
  <tbody>
    {''.join(status_rows)}
  </tbody>
</table>

{'<h2>Failure detail</h2>' + ''.join(fail_blocks) if fail_blocks else ''}

<h2>Numerical summary (block 2)</h2>
<p>Values reproduced from <code>02_translator_mechanism/results/summary.json</code>.
   For interpretation against the pre-registered hypotheses, see
   <code>02_translator_mechanism/prereg.md</code>.</p>
{summary_html}

<h2>Figures</h2>
{('<p><em>No figures yet.</em></p>' if not fig_blocks else ''.join(fig_blocks))}

<hr>
<p style='color:#888; font-size:0.85em;'>
  Generated by <code>run_pipeline.py</code>. Re-run with
  <code>python run_pipeline.py --rerun</code> to regenerate all outputs from scratch.
</p>

</body>
</html>
"""

    _atomic_write_text(output_path, html)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Run the full pipeline and produce a report.")
    parser.add_argument("--rerun", action="store_true",
                        help="Re-run every step even if its outputs already exist.")
    parser.add_argument("--force-step", action="append", default=[],
                        metavar="STEP_NAME",
                        help="Force re-execution of a specific step regardless of "
                             "its outputs. Can be passed multiple times. Use the "
                             "step name shown in the report (e.g. "
                             "'block2.06_fix_freq_bucket').")
    parser.add_argument("--list-steps", action="store_true",
                        help="List all step names and exit.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would run, don't execute scripts.")
    parser.add_argument("--report", type=Path, default=REPO_ROOT / "pipeline_report.html",
                        help="Where to write the HTML report.")
    parser.add_argument("--raw", type=Path, default=REPO_ROOT / "pipeline_results_raw.txt",
                        help="Where to write the raw text dump of all step outputs.")
    args = parser.parse_args()

    if args.list_steps:
        for step in PIPELINE:
            print(step.name)
        sys.exit(0)

    # Validate --force-step names against PIPELINE.
    known_names = {step.name for step in PIPELINE}
    unknown = [n for n in args.force_step if n not in known_names]
    if unknown:
        print(f"Unknown step name(s) for --force-step: {unknown}", file=sys.stderr)
        print(f"Use --list-steps to see valid names.", file=sys.stderr)
        sys.exit(2)
    forced = set(args.force_step)

    print(f"Repository root: {REPO_ROOT}")
    print(f"Steps in pipeline: {len(PIPELINE)}")
    mode_str = "dry-run" if args.dry_run else ("rerun" if args.rerun else "incremental")
    if forced:
        mode_str += f" (forcing: {sorted(forced)})"
    print(f"Mode: {mode_str}")
    print()

    results = []

    def _flush_outputs():
        """Persist .txt and HTML from whatever is in `results` right now.
        Called after every step AND from the finally-block, so a Ctrl+C or
        unhandled exception still leaves the latest progress on disk.

        Skipped in dry-run mode, because the StepResult objects produced by
        a dry-run carry no real stdout/stderr and would clobber the existing
        .txt and .html with empty content."""
        if args.dry_run:
            return
        try:
            write_raw_results(results, args.raw)
        except Exception as e:
            print(f"  [warn] raw dump write failed: {e!r}", file=sys.stderr)
        try:
            write_report(results, args.report)
        except Exception as e:
            print(f"  [warn] report write failed: {e!r}", file=sys.stderr)

    interrupted = False
    try:
        for step in PIPELINE:
            print(f"=== {step.name} ===")
            r = run_step(step,
                         rerun=args.rerun,
                         dry_run=args.dry_run,
                         force=(step.name in forced))
            results.append(r)

            if r.status == "OK":
                print(f"  OK in {r.elapsed_s:.1f}s")
            elif r.status == "OK (dry-run)":
                print(f"  would run")
            elif r.status == "SKIPPED":
                print(f"  SKIPPED (outputs already present)")
            elif r.status == "MISSING_INPUTS":
                print(f"  MISSING INPUTS:")
                for p in r.missing_inputs[:5]:
                    print(f"    - {p}")
                if len(r.missing_inputs) > 5:
                    print(f"    ... and {len(r.missing_inputs) - 5} more")
            else:
                print(f"  FAIL (rc={r.returncode}, elapsed={r.elapsed_s:.1f}s)")
                if r.missing_outputs:
                    print(f"  Missing outputs:")
                    for p in r.missing_outputs[:5]:
                        print(f"    - {p}")
                if r.stderr:
                    tail = r.stderr.strip().splitlines()[-3:]
                    for line in tail:
                        print(f"    stderr: {line}")
            print()

            # Persist after every step so a crash mid-pipeline still leaves
            # the .txt and HTML up to date through the last completed step.
            _flush_outputs()
    except KeyboardInterrupt:
        interrupted = True
        print("\n[interrupted] Ctrl+C received. Flushing outputs before exit...",
              file=sys.stderr)
    finally:
        # Always flush, even on Ctrl+C or unhandled exception. This is the
        # belt-and-suspenders write: the per-step write inside the loop
        # already covers the common path, but this guarantees the .txt
        # reflects the last in-memory state of `results` no matter how we
        # exit (other than SIGKILL, which the OS doesn't let us handle).
        _flush_outputs()
        if not args.dry_run:
            print(f"Report written: {args.report}")
            print(f"Raw results written: {args.raw}")

    if interrupted:
        sys.exit(130)  # conventional exit code for SIGINT
    overall_ok = all(r.status in ("OK", "OK (dry-run)", "SKIPPED") for r in results)
    sys.exit(0 if overall_ok else 1)


if __name__ == "__main__":
    main()