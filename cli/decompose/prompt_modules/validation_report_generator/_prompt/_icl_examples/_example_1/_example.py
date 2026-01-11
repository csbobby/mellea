# ruff: noqa: W293
from .._types import ICLExample

constraint_requirement = """The answer must contain exactly 3 bullet points, each starting with "- "."""

subtask_prompt = """Write three key takeaways about healthy sleep habits. Output exactly three bullet points."""

subtask_result = """- Keep a consistent sleep schedule even on weekends.
- Avoid caffeine and heavy meals late in the day.
- Create a dark, quiet, and cool sleeping environment."""

validation_report = """<validation_report>{
  "is_valid": true,
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