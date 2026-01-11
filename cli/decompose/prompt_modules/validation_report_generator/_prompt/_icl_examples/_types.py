from typing import TypedDict


class ICLExample(TypedDict):
    constraint_requirement: str
    subtask_prompt: str
    subtask_result: str
    validation_report: str  # full assistant output: <validation_report>...</validation_report> + fixed sentence