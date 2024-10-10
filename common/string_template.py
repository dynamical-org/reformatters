import string
from typing import Any


class PassThroughIfMissingDict(dict[str, Any]):
    """
    Dictionary which returns the f"${key}" as the value if the key is not present in the dictionary.
    Useful with string.Template when you want to have strings that look like $XXX but are not replaced.
    """

    def __missing__(self, key: str) -> str:
        return f"${key}"


def substitute(template_path: str, substitutions: dict[str, Any]) -> str:
    with open(template_path) as template_file:
        template = string.Template(template_file.read())
        return template.safe_substitute(PassThroughIfMissingDict(substitutions))
