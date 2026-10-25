from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "recsys-artifact.yml"


def test_workflow_targets_repository_root_and_validates_canonical_artifact():
    workflow = WORKFLOW.read_text(encoding="utf-8")

    assert '      - "**"' in workflow
    assert "recsys_corrected_artifact/**" not in workflow
    assert "working-directory: recsys_corrected_artifact" not in workflow
    assert (
        "python -m recsys_corrected_artifact.validate_protocol results/archive/v4/kion_50k_protocol_v4.json"
        in workflow
    )
