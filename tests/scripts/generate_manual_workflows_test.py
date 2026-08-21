from pathlib import Path

from scripts.generate_manual_workflows import update_choice_options


def test_update_choice_options_preserves_workflow(tmp_path: Path) -> None:
    workflow_path = tmp_path / "workflow.yml"
    workflow_path.write_text(
        """name: Manual
'on':
  workflow_dispatch:
    inputs:
      dataset_id:
        type: choice
        options:
        - old-dataset
      operation:
        type: choice
        options:
        - create-new-store
jobs:
  run:
    steps:
    - uses: actions/checkout@updated-by-dependabot
"""
    )

    update_choice_options(workflow_path, "dataset_id", ["first", "second"])

    assert (
        workflow_path.read_text()
        == """name: Manual
'on':
  workflow_dispatch:
    inputs:
      dataset_id:
        type: choice
        options:
        - first
        - second
      operation:
        type: choice
        options:
        - create-new-store
jobs:
  run:
    steps:
    - uses: actions/checkout@updated-by-dependabot
"""
    )
