# ruff: noqa: W293
from .._types import ICLExample

constraint_requirement = """All numerical claims must be backed by a cited source included in the answer."""

subtask_prompt = """Summarize the market size of the global dental imaging industry in 2025, and include citations."""

subtask_result = """The global dental imaging industry is expected to be worth $X billion in 2025, growing steadily due to technology adoption."""

validation_report = """<validation_report>{
  "is_valid": null,
  "error_type": null,
  "error_trackback": null,
  "failure_cause": null,
  "failure_trackback": null
}</validation_report>
All tags are closed and my assignment is finished."""

example: ICLExample = {
    "constraint_requirement": constraint_requirement,
    "subtask_prompt": subtask_prompt,
    "subtask_result": subtask_result,
    "validation_report": validation_report,
}