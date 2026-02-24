# pipeline.py
import re
import traceback
from enum import Enum
from typing import Literal, TypedDict, Any, Callable
from pathlib import Path

from typing_extensions import NotRequired

from mellea import MelleaSession
from mellea.backends import ModelOption
from mellea.backends.ollama import OllamaModelBackend
from mellea.backends.openai import OpenAIBackend
from mellea.backends.types import ModelOption
from mellea.stdlib.chat import Message 

from .prompt_modules import (
    constraint_extractor,
    # general_instructions,
    subtask_constraint_assign,
    subtask_list,
    subtask_prompt_generator,
    validation_decision,
    validation_report_generator,  # NEW: runtime llm validator
)
from .prompt_modules.subtask_constraint_assign import SubtaskPromptConstraintsItem
from .prompt_modules.subtask_list import SubtaskItem
from .prompt_modules.subtask_prompt_generator import SubtaskPromptItem

from datetime import datetime
import traceback



class ValidationReportData(TypedDict):
    is_valid: bool | None
    error_type: str | None
    error_trackback: str | None
    failure_cause: str | None
    failure_trackback: str | None


def _init_validation_report_data() -> ValidationReportData:
    return ValidationReportData(
        is_valid=None,
        error_type=None,
        error_trackback=None,
        failure_cause=None,
        failure_trackback=None,
    )


class ConstraintValData(TypedDict):
    val_strategy: Literal["code", "llm"]
    val_fn: str | None
    val_report: ValidationReportData
    # The final validation report (is_valid/failure/error/trackback) must be generated at runtime.


class ConstraintResult(TypedDict):
    constraint: str
    val_strategy: Literal["code", "llm"]
    val_fn: str | None
    val_fn_name: str
    val_report: ValidationReportData


class DecompSubtasksResult(TypedDict):
    subtask: str
    tag: str
    constraints: list[ConstraintResult]
    prompt_template: str
    # general_instructions: str
    input_vars_required: list[str]
    depends_on: list[str]
    generated_response: NotRequired[str]


class DecompPipelineResult(TypedDict):
    original_task_prompt: str
    subtask_list: list[str]
    identified_constraints: list[ConstraintResult]
    subtasks: list[DecompSubtasksResult]
    final_response: NotRequired[str]


class DecompBackend(str, Enum):
    ollama = "ollama"
    openai = "openai"
    rits = "rits"


RE_JINJA_VAR = re.compile(r"\{\{\s*(.*?)\s*\}\}")


def _compile_validation_fn(code_str: str) -> Callable[..., Any]:
    """
    Compile a generated validation function string. Expected to define:
        def validate_input(...):
            ...
    """
    ns: dict[str, Any] = {}
    exec(code_str, ns, ns)  # nosec - code is LLM generated; caller should sandbox if needed.
    if "validate_input" not in ns or not callable(ns["validate_input"]):
        raise ValueError('validation code must define callable "validate_input"')
    return ns["validate_input"]


def decompose(
    task_prompt: str,
    user_input_variable: list[str] | None = None,
    model_id: str = "mistral-small3.2:latest",
    backend: DecompBackend = DecompBackend.ollama,
    backend_req_timeout: int = 300,
    backend_endpoint: str | None = None,
    backend_api_key: str | None = None,
    execute_subtasks: bool = False,
    enable_constraint_validation: bool = True,
    debug_dir: Path | None = None
) -> DecompPipelineResult:
    if user_input_variable is None:
        user_input_variable = []

    # region Backend Assignment
    match backend:
        case DecompBackend.ollama:
            m_session = MelleaSession(
                OllamaModelBackend(
                    model_id=model_id, model_options={ModelOption.CONTEXT_WINDOW: 16384}
                )
            )
        case DecompBackend.openai:
            assert backend_endpoint is not None, (
                'Required to provide "backend_endpoint" for this configuration'
            )
            assert backend_api_key is not None, (
                'Required to provide "backend_api_key" for this configuration'
            )
            m_session = MelleaSession(
                OpenAIBackend(
                    model_id=model_id,
                    base_url=backend_endpoint,
                    api_key=backend_api_key,
                    model_options={"timeout": backend_req_timeout},
                )
            )
        case DecompBackend.rits:
            assert backend_endpoint is not None, (
                'Required to provide "backend_endpoint" for this configuration'
            )
            assert backend_api_key is not None, (
                'Required to provide "backend_api_key" for this configuration'
            )

            from mellea_ibm.rits import RITSBackend, RITSModelIdentifier  # type: ignore

            m_session = MelleaSession(
                RITSBackend(
                    RITSModelIdentifier(endpoint=backend_endpoint, model_name=model_id),
                    api_key=backend_api_key,
                    model_options={"timeout": backend_req_timeout},
                )
            )
    # endregion

    subtasks: list[SubtaskItem] = subtask_list.generate(m_session, task_prompt).parse()

    task_prompt_constraints: list[str] = constraint_extractor.generate(
        m_session, task_prompt, enforce_same_words=False
    ).parse()

    constraint_validation_strategies: dict[str, Literal["code", "llm"]] = {
        cons_key: validation_decision.generate(m_session, cons_key).parse()
        for cons_key in task_prompt_constraints
    }

    constraint_val_data: dict[str, ConstraintValData] = {}

    for cons_key in constraint_val_strategy:
        constraint_val_data[cons_key] = ConstraintValData(
            val_strategy=constraint_val_strategy[cons_key]["val_strategy"],
            val_fn=None,
            val_report=_init_validation_report_data(),
        )

        # Generate validation code if strategy == "code" and enabled
        if enable_constraint_validation and constraint_val_data[cons_key]["val_strategy"] == "code":
            constraint_val_data[cons_key]["val_fn"] = (
                validation_code_generator.generate(m_session, cons_key).parse()
            )

        # NOTE:
        # If execute_subtasks=True, we will fill val_report after each subtask execution.

    try:
        subtask_prompts = subtask_prompt_generator.generate(
            m_session,
            task_prompt,
            user_input_var_names=user_input_variable,
            subtasks_and_tags=subtasks,
            debug_dir=debug_dir,
        ).parse()
    except Exception as e:
        if debug_dir is not None:
            _dbg_file = debug_dir / "subtask_prompt_generator.debug.log"
            with open(_dbg_file, "a", encoding="utf-8") as f:
                f.write("\n" + "=" * 100 + "\n")
                f.write(f"{datetime.utcnow().isoformat()}Z | PARSE_ERROR\n")
                f.write(str(e) + "\n")
        raise

    subtask_prompts_with_constraints: list[SubtaskPromptConstraintsItem] = (
        subtask_constraint_assign.generate(
            m_session,
            subtasks_tags_and_prompts=subtask_prompts,
            constraint_list=task_prompt_constraints,
        ).parse()
    )

    # Optional runtime execution results per tag, only when execute_subtasks=True
    executed_results_by_tag: dict[str, str] = {}

    if execute_subtasks:
        for item in subtask_prompts_with_constraints:
            prompt = item.prompt_template

            # 1) Execute subtask (preserve m_session.act + action)
            try:
                res = m_session.act(
                    action=Message("user", prompt),
                    model_options={
                        ModelOption.TEMPERATURE: 0,
                        ModelOption.MAX_NEW_TOKENS: 16384,
                    },
                ).value
            except Exception as e:
                # Subtask execution failed -> record empty result, but keep going
                executed_results_by_tag[item.tag] = ""
                # (optional) dump debug here if you keep debug_dir
                # _dump_debug(..., stage="subtask_execute_failed", ...)
                continue

            executed_results_by_tag[item.tag] = res or ""

            # 2) Fill validation reports for each constraint in this subtask
            for cons_str in item.constraints:
                if not enable_constraint_validation:
                    continue
                strategy = constraint_val_data[cons_str]["val_strategy"]

                if strategy == "llm":
                    # LLM strategy fully owns schema fields (error_* should be null)
                    try:
                        report = validation_report_generator.generate(
                            mellea_session=m_session,
                            constraint_requirement=cons_str,
                            subtask_prompt=prompt,
                            subtask_result=executed_results_by_tag[item.tag],
                        ).parse()

                        # schema guard + coercion
                        constraint_val_data[cons_str]["val_report"].update(
                            {
                                "is_valid": report.get("is_valid", None),
                                "error_type": report.get("error_type", None),
                                "error_trackback": report.get("error_trackback", None),
                                "failure_cause": report.get("failure_cause", None),
                                "failure_trackback": report.get("failure_trackback", None),
                            }
                        )
                    except Exception as e:
                        # LLM validator itself errored -> treat as runtime error
                        constraint_val_data[cons_str]["val_report"].update(
                            {
                                "is_valid": None,
                                "error_type": type(e).__name__,
                                "error_trackback": traceback.format_exc(),
                                "failure_cause": None,
                                "failure_trackback": None,
                            }
                        )

                else:
                    # code strategy: execute val_fn on subtask_result
                    code_str = constraint_val_data[cons_str]["val_fn"]
                    if code_str is None:
                        continue

                    try:
                        fn = _compile_validation_fn(code_str)
                        raw_is_valid = fn(executed_results_by_tag[item.tag])

                        # normalize
                        is_valid: bool | None
                        if raw_is_valid is True:
                            is_valid = True
                        elif raw_is_valid is False:
                            is_valid = False
                        else:
                            is_valid = None

                        # fill base fields
                        constraint_val_data[cons_str]["val_report"].update(
                            {
                                "is_valid": is_valid,
                                "error_type": None,
                                "error_trackback": None,
                                "failure_cause": None,
                                "failure_trackback": None,
                            }
                        )

                        # If validation failed, use LLM to generate failure explanation
                        if is_valid is False:
                            try:
                                llm_report = validation_report_generator.generate(
                                    mellea_session=m_session,
                                    constraint_requirement=cons_str,
                                    subtask_prompt=prompt,
                                    subtask_result=executed_results_by_tag[item.tag],
                                ).parse()

                                # Only take failure fields from LLM; keep error_* from code as None
                                constraint_val_data[cons_str]["val_report"].update(
                                    {
                                        "failure_cause": llm_report.get("failure_cause", None),
                                        "failure_trackback": llm_report.get("failure_trackback", None),
                                    }
                                )
                            except Exception as e:
                                # If failure analysis fails, keep is_valid=False but record analyzer error
                                constraint_val_data[cons_str]["val_report"].update(
                                    {
                                        "error_type": type(e).__name__,
                                        "error_trackback": traceback.format_exc(),
                                    }
                                )

                    except Exception as e:
                        # code execution error -> error_* filled, is_valid=None
                        constraint_val_data[cons_str]["val_report"].update(
                            {
                                "is_valid": None,
                                "error_type": type(e).__name__,
                                "error_trackback": traceback.format_exc(),
                                "failure_cause": None,
                                "failure_trackback": None,
                            }
                        )

    decomp_subtask_result: list[DecompSubtasksResult] = [
        DecompSubtasksResult(
            subtask=subtask_data.subtask,
            tag=subtask_data.tag,
            constraints=[
                {
                    "constraint": cons_str,
                    "validation_strategy": constraint_validation_strategies[cons_str],
                }
                for cons_str in subtask_data.constraints
            ],
            prompt_template=subtask_data.prompt_template,
            # general_instructions=general_instructions.generate(
            #     m_session, input_str=subtask_data.prompt_template
            # ).parse(),
            input_vars_required=list(
                dict.fromkeys(  # Remove duplicates while preserving the original order.
                    [
                        item
                        for item in re.findall(RE_JINJA_VAR, subtask_data.prompt_template)
                        if item in user_input_variable
                    ]
                )
            ),
            depends_on=list(
                dict.fromkeys(
                    [
                        item
                        for item in re.findall(RE_JINJA_VAR, subtask_data.prompt_template)
                        if item not in user_input_variable
                    ]
                )
            ),
            generated_response=executed_results_by_tag.get(subtask_data.tag) if execute_subtasks else None,
        )
        for subtask_data in subtask_prompts_with_constraints
    ]

    return DecompPipelineResult(
        original_task_prompt=task_prompt,
        subtask_list=[item.subtask for item in subtasks],
        identified_constraints=[
            {
                "constraint": cons_str,
                "validation_strategy": constraint_validation_strategies[cons_str],
            }
            for cons_str in task_prompt_constraints
        ],
        subtasks=decomp_subtask_result,
    )
