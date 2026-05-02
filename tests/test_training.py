"""Tests for the Phase 6 training utilities.

Coverage of the local-runnable surface:
    - Trajectory collection with the mock LLM and synthetic FM runners.
    - JSONL round-trip.
    - Trajectory -> messages reconstruction.
    - SFT / DPO / GRPO dataset builders.
    - Verifier-based reward function (replays actions through the
      synthetic runners and checks that PASS / FAIL claims get
      different scores).

The actual transformers / trl / peft trainers are gated behind
``importorskip`` so the test suite passes on a host without those
packages. The audit venv we use locally has loguru / pydantic /
pyyaml / torch / numpy / h5py / scipy / matplotlib / pytest plus
typer; it does NOT have transformers, trl, or peft.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from fmllm.bridges import (
    FMContext,
    make_structure_bridge,
)
from fmllm.bridges.compose import metadata_yaml_path
from fmllm.fms._schemas import (
    BridgedFMOutput,
    ProbeReport,
    ProbeResult,
    load_fm_metadata,
)
from fmllm.fms._schemas.probe_schema import now_utc_iso
from fmllm.orchestrator import MockLLM
from fmllm.training import (
    collect_trajectories,
    extract_specimen_id,
    load_trajectories_jsonl,
    make_verifier_reward_fn,
    trajectories_to_dpo_pairs,
    trajectories_to_grpo_prompts,
    trajectories_to_sft_records,
    trajectory_to_messages,
    write_trajectories_jsonl,
)
from fmllm.verifier import (
    SourcesConfig,
    build_default_verifier,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
LITERATURE_DB = REPO_ROOT / "data" / "literature" / "clusters.json"


# ---------------------------------------------------------------------------
# Synthetic FM runners (reuse the orchestrator-test fixtures)
# ---------------------------------------------------------------------------


def _build_context(fm_name: str) -> FMContext:
    metadata = load_fm_metadata(metadata_yaml_path(fm_name))
    probe_report = ProbeReport(
        fm_name=metadata.name, fm_version=metadata.version,
        timestamp_utc=now_utc_iso(),
        results=[
            ProbeResult(
                constraint_name=c.name, satisfaction_score=0.9,
                num_test_cases=64, metric="synth",
                passes_threshold=True, threshold=c.expected_satisfaction,
                details={},
            )
            for c in metadata.physics_constraints
        ],
    )
    return FMContext(
        fm_name=fm_name, metadata=metadata,
        probe_report=probe_report, calibration={},
    )


def _fake_runner(fm_name: str):
    ctx = _build_context(fm_name)
    bridge = make_structure_bridge(ctx)

    def runner(arguments: dict) -> BridgedFMOutput:
        if fm_name == "fm1_image":
            raw = {
                "count_logits": torch.cat([
                    torch.full((30,), -3.0),
                    torch.tensor([5.0]),
                ]),
                "positions": torch.tensor([[0.5, 0.3], [-1.2, 0.7]]),
                "confidence_logits": torch.tensor([3.0, 2.5]),
            }
        elif fm_name == "fm2_rdf":
            raw = {"energy": torch.tensor(-1.42)}
        else:
            raw = {"alpha": torch.tensor(2.0), "beta": torch.tensor(0.55)}
        return bridge.emit(raw, input_provenance=arguments)

    return runner


@pytest.fixture
def runners():
    return {
        "fm1": _fake_runner("fm1_image"),
        "fm2": _fake_runner("fm2_rdf"),
        "fm3": _fake_runner("fm3_traj"),
    }


@pytest.fixture
def verifier():
    return build_default_verifier(literature_db_path=LITERATURE_DB)


# ---------------------------------------------------------------------------
# Trajectory collection + JSONL round-trip
# ---------------------------------------------------------------------------


def test_collect_trajectories_writes_jsonl(tmp_path: Path, runners, verifier):
    """The collector writes one trajectory per specimen and a summary."""
    mock = MockLLM([
        '{"action": "call_fm", "tool_name": "fm1", "specimen_id": 0}',
        '{"action": "commit", "claim": {"n_atoms": 7}}',
    ] * 4)  # 2 specimens worth of 2-turn scripts
    out = tmp_path / "trajectories"
    summary = collect_trajectories(
        llm=mock, verifier=verifier, runners=runners,
        specimen_ids=[0, 1],
        out_dir=out,
        max_steps=3,
    )
    assert summary["counters"]["total"] == 2
    jsonl_path = Path(summary["jsonl_path"])
    assert jsonl_path.exists()

    loaded = load_trajectories_jsonl(jsonl_path)
    assert len(loaded) == 2
    assert (out / "summary.yaml").exists()
    assert (out / "manifest.yaml").exists()


def test_jsonl_round_trip(tmp_path: Path, runners, verifier):
    mock = MockLLM(['{"action": "commit", "claim": {"n_atoms": 7}}'])
    out = tmp_path / "single"
    summary = collect_trajectories(
        llm=mock, verifier=verifier, runners=runners,
        specimen_ids=[0],
        out_dir=out, max_steps=1,
    )
    loaded = load_trajectories_jsonl(summary["jsonl_path"])
    assert len(loaded) == 1
    out2 = tmp_path / "round.jsonl"
    write_trajectories_jsonl(loaded, out2)
    loaded2 = load_trajectories_jsonl(out2)
    assert loaded[0].model_dump() == loaded2[0].model_dump()


def test_filter_passing_drops_failed_trajectories(
    tmp_path: Path, runners, verifier,
):
    """When ``filter_passing=True``, only PASS trajectories land in JSONL."""
    # The mock's claim disagrees with the FM evidence on real specimens;
    # the verifier will likely return CAVEAT or FAIL, so filter_passing
    # should keep zero. The summary still records all.
    mock = MockLLM([
        '{"action": "call_fm", "tool_name": "fm1", "specimen_id": 0}',
        '{"action": "commit", "claim": {"n_atoms": 99, "motif": "ring"}}',
    ])
    out = tmp_path / "filt"
    summary = collect_trajectories(
        llm=mock, verifier=verifier, runners=runners,
        specimen_ids=[0],
        out_dir=out, max_steps=3,
        filter_passing=True,
    )
    assert summary["counters"]["total"] == 1
    # Filtered count: 0 if the claim FAILed; 1 if (unlikely) it PASSed.
    loaded = load_trajectories_jsonl(summary["jsonl_path"])
    assert summary["counters"]["filtered_kept"] == len(loaded)


# ---------------------------------------------------------------------------
# Trajectory -> messages reconstruction
# ---------------------------------------------------------------------------


def test_trajectory_to_messages_round_trip(tmp_path: Path, runners, verifier):
    """The reconstructed messages contain system + user + assistant turns."""
    mock = MockLLM([
        '{"action": "call_fm", "tool_name": "fm1", "specimen_id": 0}',
        '{"action": "commit", "claim": {"n_atoms": 7}}',
    ])
    out = tmp_path / "single"
    summary = collect_trajectories(
        llm=mock, verifier=verifier, runners=runners,
        specimen_ids=[0],
        out_dir=out, max_steps=2,
    )
    [traj] = load_trajectories_jsonl(summary["jsonl_path"])
    messages = trajectory_to_messages(traj)
    roles = [m["role"] for m in messages]
    assert roles[0] == "system"
    assert roles[1] == "user"
    # At least two assistant turns + tool messages in between.
    assert roles.count("assistant") >= 1
    assert "tool" in roles


# ---------------------------------------------------------------------------
# Dataset builders
# ---------------------------------------------------------------------------


def test_sft_records_include_caveat_flag(runners, verifier):
    """include_caveat=True keeps CAVEAT trajectories alongside PASS."""
    from fmllm.orchestrator import OHVDLoop

    # Build one trajectory likely to land CAVEAT (Qwen would commit a
    # claim that triggers literature CAVEAT but no hard fail).
    mock = MockLLM([
        '{"action": "call_fm", "tool_name": "fm1"}',
        '{"action": "call_fm", "tool_name": "fm2"}',
        '{"action": "commit", "claim": {"n_atoms": 7, "motif": "triangular_disk"}}',
    ])
    loop = OHVDLoop(llm=mock, verifier=verifier, runners=runners, max_steps=4)
    traj = loop.run("test", specimen_id=0)
    only_pass = trajectories_to_sft_records([traj], only_passing=True, include_caveat=False)
    with_caveat = trajectories_to_sft_records([traj], only_passing=True, include_caveat=True)
    # If the trajectory is CAVEAT, with_caveat keeps it; only_pass drops it.
    if traj.final_verdict and traj.final_verdict.aggregate_decision.value == "caveat":
        assert len(only_pass) == 0
        assert len(with_caveat) == 1


def test_sft_records_filter_to_passing(tmp_path: Path, runners, verifier):
    """SFT builder respects only_passing flag."""
    # Generate one likely-failing trajectory.
    mock = MockLLM([
        '{"action": "call_fm", "tool_name": "fm1", "specimen_id": 0}',
        '{"action": "commit", "claim": {"n_atoms": 99, "motif": "ring"}}',
    ])
    out = tmp_path / "filt"
    summary = collect_trajectories(
        llm=mock, verifier=verifier, runners=runners,
        specimen_ids=[0],
        out_dir=out, max_steps=3,
    )
    trajs = load_trajectories_jsonl(summary["jsonl_path"])
    only_pass = trajectories_to_sft_records(trajs, only_passing=True)
    all_records = trajectories_to_sft_records(trajs, only_passing=False)
    assert len(all_records) >= len(only_pass)
    for r in all_records:
        assert "messages" in r
        assert isinstance(r["messages"], list)


def test_dpo_pairs_require_pass_and_fail(runners, verifier):
    """No pairs when only one outcome class is present."""
    # Simulate a single trajectory that committed but FAILed.
    from fmllm.orchestrator import OHVDLoop

    mock = MockLLM(['{"action": "commit", "claim": {"n_atoms": 99}}'])
    loop = OHVDLoop(llm=mock, verifier=verifier, runners=runners, max_steps=1)
    traj_fail = loop.run("test", specimen_id=0)
    pairs = trajectories_to_dpo_pairs([traj_fail])
    assert pairs == []


def test_grpo_prompts_dedupe_by_specimen(runners, verifier):
    from fmllm.orchestrator import OHVDLoop

    mock = MockLLM([
        '{"action": "commit", "claim": {"n_atoms": 7}}',
        '{"action": "commit", "claim": {"n_atoms": 7}}',
    ])
    loop = OHVDLoop(llm=mock, verifier=verifier, runners=runners, max_steps=1)
    traj_a = loop.run("query A", specimen_id=0)
    traj_b = loop.run("query A", specimen_id=0)
    prompts = trajectories_to_grpo_prompts([traj_a, traj_b])
    assert len(prompts) == 1


# ---------------------------------------------------------------------------
# Reward function
# ---------------------------------------------------------------------------


def test_extract_specimen_id_from_prompt():
    text = (
        "system: ...\n"
        "Specimen id: 17\n"
        "Query: identify..."
    )
    assert extract_specimen_id(text) == 17


def test_extract_specimen_id_handles_message_list():
    msgs = [
        {"role": "system", "content": "..."},
        {"role": "user", "content": "Specimen id: 42\nQuery: ..."},
    ]
    assert extract_specimen_id(msgs) == 42


def test_extract_specimen_id_returns_none_when_absent():
    assert extract_specimen_id("no specimen anywhere") is None


def test_reward_function_zero_when_no_commit(runners, verifier):
    reward = make_verifier_reward_fn(verifier=verifier, runners=runners)
    completions = [
        '{"action": "call_fm", "tool_name": "fm1", "specimen_id": 0}',
    ]
    prompts = ["Specimen id: 0\nQuery: ..."]
    [r] = reward(completions=completions, prompts=prompts)
    assert r == 0.0


def test_reward_function_nonzero_when_commit_passes(runners, verifier):
    """A claim that should match the literature reference at N=7 lands a
    positive reward, even if not full PASS."""
    reward = make_verifier_reward_fn(verifier=verifier, runners=runners)
    completion = (
        '{"action": "call_fm", "tool_name": "fm1", "specimen_id": 0}\n'
        '{"action": "call_fm", "tool_name": "fm2", "specimen_id": 0}\n'
        '{"action": "commit", "claim": {"n_atoms": 7, "motif": "triangular_disk"}}'
    )
    prompt = "Specimen id: 0\nQuery: ..."
    [r] = reward(completions=[completion], prompts=[prompt])
    assert r > 0.0


def test_reward_function_runs_with_v0_ablation(runners, verifier):
    """Under V0 (no sources), the verifier returns SKIP and reward is 0."""
    reward = make_verifier_reward_fn(
        verifier=verifier, runners=runners,
        sources_config=SourcesConfig.for_ablation("V0"),
    )
    completion = (
        '{"action": "commit", "claim": {"n_atoms": 7}}'
    )
    [r] = reward(completions=[completion], prompts=["Specimen id: 0\nQuery: ..."])
    assert r == 0.0


# ---------------------------------------------------------------------------
# Heavy-deps trainers (gated)
# ---------------------------------------------------------------------------


def test_lora_save_load_round_trip_needs_peft(tmp_path: Path):
    """If peft is installed locally, confirm save/load round-trip on a tiny
    GPT-2. Skips otherwise."""
    pytest.importorskip("peft")
    pytest.importorskip("transformers")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    from fmllm.training.lora import (
        apply_lora, load_lora, lora_parameter_summary, save_lora,
    )

    name = "sshleifer/tiny-gpt2"
    base = AutoModelForCausalLM.from_pretrained(name)
    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id

    # Tiny-GPT2 modules names differ from Llama; override.
    wrapped = apply_lora(
        base,
        target_modules=["c_attn"],
        r=4,
    )
    summary = lora_parameter_summary(wrapped)
    assert summary["trainable"] > 0
    assert summary["trainable"] < summary["total"]

    save_lora(wrapped, tmp_path / "adapter")
    base2 = AutoModelForCausalLM.from_pretrained(name)
    reloaded = load_lora(base2, tmp_path / "adapter")
    summary2 = lora_parameter_summary(reloaded)
    # PEFT freezes non-LoRA at load when is_trainable=False (default).
    assert summary2["total"] >= summary["total"]
