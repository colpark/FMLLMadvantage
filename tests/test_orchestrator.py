"""Tests for the orchestrator: trajectory schemas, action parser, and OHVD loop.

The tests use :class:`MockLLM` to drive the loop deterministically.
The real :class:`TransformersLLM` lives behind a lazy import and never
loads here.
"""

from __future__ import annotations

import json

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
from fmllm.orchestrator import (
    ActionType,
    LLMAction,
    MockLLM,
    OHVDLoop,
    StepType,
    TerminationReason,
    Trajectory,
    parse_llm_response,
)
from fmllm.verifier import (
    PhysicalStateClaim,
    SourcesConfig,
    build_default_verifier,
)


# ---------------------------------------------------------------------------
# parse_llm_response
# ---------------------------------------------------------------------------


def test_parse_call_fm_action():
    action = parse_llm_response(
        '{"action": "call_fm", "tool_name": "fm1", "specimen_id": 42}',
    )
    assert action.action_type is ActionType.CALL_FM
    assert action.tool_call is not None
    assert action.tool_call.tool_name == "fm1"
    assert action.tool_call.arguments == {"specimen_id": 42}


def test_parse_hypothesize_action():
    action = parse_llm_response(
        '{"action": "hypothesize", "claim": {"n_atoms": 7, "temperature": 0.5}}',
    )
    assert action.action_type is ActionType.HYPOTHESIZE
    assert action.claim is not None
    assert action.claim.n_atoms == 7
    assert action.claim.temperature == 0.5


def test_parse_commit_action():
    action = parse_llm_response(
        '{"action": "commit", "claim": {"n_atoms": 7, "motif": "triangular_disk"}}',
    )
    assert action.action_type is ActionType.COMMIT
    assert action.claim is not None
    assert action.claim.motif == "triangular_disk"


def test_parse_handles_leading_prose():
    """Real LLMs prepend prose before the JSON; parser must tolerate that."""
    response = (
        "I will start by calling FM1 to identify the atoms.\n"
        '{"action": "call_fm", "tool_name": "fm1", "specimen_id": 7}'
    )
    action = parse_llm_response(response)
    assert action.action_type is ActionType.CALL_FM
    assert action.tool_call.tool_name == "fm1"


def test_parse_rejects_empty_response():
    action = parse_llm_response("")
    assert action.action_type is ActionType.ERROR


def test_parse_rejects_non_json_text():
    action = parse_llm_response("just talking, no json")
    assert action.action_type is ActionType.ERROR


def test_parse_rejects_unknown_action():
    action = parse_llm_response('{"action": "do_something"}')
    assert action.action_type is ActionType.ERROR


def test_parse_rejects_malformed_json():
    action = parse_llm_response('{"action": "call_fm", "tool_name": "fm1"')
    assert action.action_type is ActionType.ERROR


# ---------------------------------------------------------------------------
# Mock LLM behavior
# ---------------------------------------------------------------------------


def test_mock_llm_returns_scripted_responses_in_order():
    mock = MockLLM([
        '{"action": "call_fm", "tool_name": "fm1", "specimen_id": 7}',
        '{"action": "commit", "claim": {"n_atoms": 7}}',
    ])
    msgs = [{"role": "system", "content": "x"}]
    assert mock.chat(msgs) == '{"action": "call_fm", "tool_name": "fm1", "specimen_id": 7}'
    assert mock.chat(msgs) == '{"action": "commit", "claim": {"n_atoms": 7}}'


def test_mock_llm_emits_error_when_exhausted():
    mock = MockLLM([])
    out = mock.chat([])
    assert json.loads(out)["action"] == "error"


# ---------------------------------------------------------------------------
# Loop fixtures
# ---------------------------------------------------------------------------


def _build_context(fm_name: str) -> FMContext:
    metadata = load_fm_metadata(metadata_yaml_path(fm_name))
    probe_report = ProbeReport(
        fm_name=metadata.name,
        fm_version=metadata.version,
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
    """Return a callable that produces a static BridgedFMOutput per call."""
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
    from pathlib import Path
    repo_root = Path(__file__).resolve().parents[1]
    return build_default_verifier(
        literature_db_path=repo_root / "data" / "literature" / "clusters.json",
    )


# ---------------------------------------------------------------------------
# Loop behavior
# ---------------------------------------------------------------------------


def test_loop_terminates_on_immediate_commit(runners, verifier):
    mock = MockLLM(['{"action": "commit", "claim": {"n_atoms": 7}}'])
    loop = OHVDLoop(llm=mock, verifier=verifier, runners=runners, max_steps=4)
    traj = loop.run("test query", specimen_id=7)
    assert traj.termination is TerminationReason.COMMITTED
    assert traj.final_claim is not None
    assert traj.final_claim.n_atoms == 7
    assert traj.final_verdict is not None
    # Step types: FINAL + VERIFIER_VERDICT
    types = traj.step_types()
    assert StepType.FINAL in types
    assert StepType.VERIFIER_VERDICT in types


def test_loop_dispatches_call_fm_actions(runners, verifier):
    mock = MockLLM([
        '{"action": "call_fm", "tool_name": "fm1", "specimen_id": 7}',
        '{"action": "call_fm", "tool_name": "fm2", "specimen_id": 7}',
        '{"action": "commit", "claim": {"n_atoms": 7}}',
    ])
    loop = OHVDLoop(llm=mock, verifier=verifier, runners=runners, max_steps=10)
    traj = loop.run("test query", specimen_id=7)

    obs_steps = [s for s in traj.steps if s.step_type is StepType.OBSERVATION]
    assert len(obs_steps) == 2
    assert obs_steps[0].bridged_output.source.fm_name == "fm1_image"
    assert obs_steps[1].bridged_output.source.fm_name == "fm2_rdf"
    assert traj.termination is TerminationReason.COMMITTED


def test_loop_handles_unknown_tool_as_error_and_continues(runners, verifier):
    mock = MockLLM([
        '{"action": "call_fm", "tool_name": "fm99", "specimen_id": 7}',
        '{"action": "commit", "claim": {"n_atoms": 7}}',
    ])
    loop = OHVDLoop(llm=mock, verifier=verifier, runners=runners, max_steps=4)
    traj = loop.run("test", specimen_id=7)
    assert any(s.step_type is StepType.ERROR for s in traj.steps)
    assert traj.termination is TerminationReason.COMMITTED


def test_loop_handles_parse_error_and_continues(runners, verifier):
    mock = MockLLM([
        "this is just prose, no JSON",
        '{"action": "commit", "claim": {"n_atoms": 7}}',
    ])
    loop = OHVDLoop(llm=mock, verifier=verifier, runners=runners, max_steps=4)
    traj = loop.run("test", specimen_id=7)
    assert any(s.step_type is StepType.ERROR for s in traj.steps)
    assert traj.termination is TerminationReason.COMMITTED


def test_loop_terminates_on_budget_exhausted(runners, verifier):
    """An LLM that never commits exhausts the step budget."""
    responses = [
        '{"action": "hypothesize", "claim": {"n_atoms": 7}}'
    ] * 5
    mock = MockLLM(responses)
    loop = OHVDLoop(llm=mock, verifier=verifier, runners=runners, max_steps=3)
    traj = loop.run("test", specimen_id=7)
    assert traj.termination is TerminationReason.BUDGET_EXHAUSTED
    assert traj.final_claim is None


def test_loop_feeds_verdicts_back_to_llm_context(runners, verifier):
    mock = MockLLM([
        '{"action": "hypothesize", "claim": {"n_atoms": 7}}',
        '{"action": "commit", "claim": {"n_atoms": 7}}',
    ])
    loop = OHVDLoop(llm=mock, verifier=verifier, runners=runners, max_steps=4)
    loop.run("test", specimen_id=7)

    # Inspect the messages the mock saw on the second call. The first call
    # had only the system + user. The second call should have the assistant
    # response + a tool message with the verdict summary.
    second_call_messages = mock.received_messages[1]
    roles = [m["role"] for m in second_call_messages]
    assert "tool" in roles
    tool_payload = next(
        m for m in second_call_messages if m["role"] == "tool"
    )["content"]
    parsed = json.loads(tool_payload)
    assert "aggregate_decision" in parsed
    assert "sources" in parsed


def test_loop_passes_sources_config_through_to_verifier(runners, verifier):
    """Sources config V0 produces an aggregate-skip verdict."""
    mock = MockLLM(['{"action": "commit", "claim": {"n_atoms": 7}}'])
    loop = OHVDLoop(
        llm=mock, verifier=verifier, runners=runners, max_steps=2,
        sources_config=SourcesConfig.for_ablation("V0"),
    )
    traj = loop.run("test", specimen_id=7)
    assert traj.final_verdict is not None
    assert traj.final_verdict.aggregate_decision.value == "skip"


def test_trajectory_round_trips_through_json(runners, verifier):
    mock = MockLLM([
        '{"action": "call_fm", "tool_name": "fm2", "specimen_id": 7}',
        '{"action": "commit", "claim": {"n_atoms": 7, "temperature": 0.5}}',
    ])
    loop = OHVDLoop(llm=mock, verifier=verifier, runners=runners, max_steps=4)
    traj = loop.run("test", specimen_id=7)
    payload = traj.model_dump_json()
    rehydrated = Trajectory.model_validate_json(payload)
    assert rehydrated.model_dump() == traj.model_dump()
