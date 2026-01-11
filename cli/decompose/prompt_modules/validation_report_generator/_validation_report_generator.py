import json
import re
from collections.abc import Callable
from typing import Any, TypeVar, final

from mellea import MelleaSession
from mellea.backends.types import ModelOption
from mellea.stdlib.chat import Message

from .._prompt_modules import PromptModule, PromptModuleString
from ._exceptions import BackendGenerationError, TagExtractionError
from ._prompt import get_system_prompt, get_user_prompt

T = TypeVar("T")

RE_VALIDATION_REPORT = re.compile(
    r"<validation_report>(.+?)</validation_report>", flags=re.IGNORECASE | re.DOTALL
)


@final
class _ValidationReportGenerator(PromptModule):
    @staticmethod
    def _default_parser(generated_str: str) -> dict[str, Any]:
        r"""Default parser of the `validation_report_generator` module.

        Extracts JSON object inside <validation_report> tags and parses it.

        Expected JSON keys (schema-aligned):
        - is_valid: bool | null
        - error_type: str | null
        - error_trackback: str | null
        - failure_cause: str | null
        - failure_trackback: str | null

        Raises:
            TagExtractionError if tags not found or JSON parsing failed.
        """
        m = re.search(RE_VALIDATION_REPORT, generated_str)
        validation_report_str: str | None = m.group(1).strip() if m else None
        if validation_report_str is None:
            raise TagExtractionError(
                'LLM failed to generate correct tags for extraction: "<validation_report>"'
            )

        try:
            obj = json.loads(validation_report_str)
        except Exception as e:
            raise TagExtractionError(f"Failed to parse <validation_report> JSON: {e}")

        return obj

    def generate(
        self,
        mellea_session: MelleaSession,
        constraint_requirement: str,
        subtask_prompt: str,
        subtask_result: str,
        max_new_tokens: int = 2048,
        parser: Callable[[str], T] = _default_parser,  # type: ignore[assignment]
        **kwargs: dict[str, Any],
    ) -> PromptModuleString[T]:
        """Generates a runtime validation report for a given constraint.

        This module is used at *runtime* to decide whether a constraint is satisfied,
        based on:
          - constraint text
          - subtask prompt
          - subtask output/result

        It returns a JSON object inside <validation_report> tags.

        Args:
            constraint_requirement: natural language constraint
            subtask_prompt: the prompt used to obtain the subtask output
            subtask_result: the generated output of the subtask
        """
        system_prompt = get_system_prompt()
        user_prompt = get_user_prompt(
            constraint_requirement=constraint_requirement,
            subtask_prompt=subtask_prompt,
            subtask_result=subtask_result,
        )

        action = Message("user", user_prompt)

        try:
            gen_result = mellea_session.act(
                action=action,
                model_options={
                    ModelOption.SYSTEM_PROMPT: system_prompt,
                    ModelOption.TEMPERATURE: 0,
                    ModelOption.MAX_NEW_TOKENS: max_new_tokens,
                },
            ).value
        except Exception as e:
            raise BackendGenerationError(f"LLM generation failed: {e}")

        if gen_result is None:
            raise BackendGenerationError("LLM generation failed: value attribute is None")

        return PromptModuleString(gen_result, parser)


validation_report_generator = _ValidationReportGenerator()