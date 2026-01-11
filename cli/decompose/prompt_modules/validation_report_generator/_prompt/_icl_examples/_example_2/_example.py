# ruff: noqa: W293
from .._types import ICLExample

constraint_requirement = """Do not use any uppercase letters in the final answer."""

subtask_prompt = """Rewrite the sentence using only lowercase letters: "Hello World!" """

subtask_result = """hello world!
note: i removed the exclamation and kept it simple.
also, the quoted text "Hello World!" appears above.
"""

validation_report = """<validation_report>{
  "is_valid": false,
  "error_type": null,
  "error_trackback": null,
  "failure_cause": "The result includes uppercase letters due to the quoted text \\"Hello World!\\". This violates the requirement that no uppercase letters appear anywhere in the final answer.",
  "failure_trackback": "Uppercase-letter prohibition violated: the response includes uppercase characters from quoted content."
}</validation_report>
All tags are closed and my assignment is finished."""

example: ICLExample = {
    "constraint_requirement": constraint_requirement,
    "subtask_prompt": subtask_prompt,
    "subtask_result": subtask_result,
    "validation_report": validation_report,
}