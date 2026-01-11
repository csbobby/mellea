from pathlib import Path

from jinja2 import Environment, FileSystemLoader

from ._icl_examples import ICLExample, icl_examples as default_icl_examples

this_file_dir = Path(__file__).resolve().parent

environment = Environment(loader=FileSystemLoader(this_file_dir), autoescape=False)
system_template = environment.get_template("system_template.jinja2")
user_template = environment.get_template("user_template.jinja2")


def get_system_prompt(
    icl_examples: list[ICLExample] = default_icl_examples,
    enforce_same_words: bool = False,
) -> str:
    # NOTE:
    # "enforce_same_words" is kept for interface compatibility, even if not used by this prompt.
    return system_template.render(
        icl_examples=icl_examples, enforce_same_words=enforce_same_words
    ).strip()


def get_user_prompt(
    constraint_requirement: str,
    subtask_prompt: str,
    subtask_result: str,
) -> str:
    return user_template.render(
        constraint_requirement=constraint_requirement,
        subtask_prompt=subtask_prompt,
        subtask_result=subtask_result,
    ).strip()