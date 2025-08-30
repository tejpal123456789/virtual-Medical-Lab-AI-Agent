from .templates import get_template


def build_prompt(base: str, instruction_key: str | None = None) -> str:
    tmpl = get_template(instruction_key) if instruction_key else ""
    return f"{tmpl}\n\n{base}" if tmpl else base 