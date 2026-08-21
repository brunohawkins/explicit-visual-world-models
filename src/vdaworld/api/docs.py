import inspect
from typing import List
from vdaworld.core.api import WorldAPI

_API_INSTANCE = WorldAPI()


def get_api_documentation(method_names: list[str]) -> str:
    """
    Fetches the precise docstrings and type hints for specific WorldAPI methods.
    Call this BEFORE writing your generation code to understand the tool signatures.

    Arguments:
        method_names: A list of method names (e.g., ["segment", "estimate_3d_points", "intrinsics"]).

    Returns:
        A string containing the documentation for the requested methods.
    """
    if isinstance(method_names, str):
        method_names = [method_names]

    docs = []

    print(
        f"[*] Native Tool Call Invoked! LLM requested documentation for: {method_names}"
    )

    # Get all public methods of WorldAPI
    available_methods = {
        name: func
        for name, func in inspect.getmembers(_API_INSTANCE, predicate=inspect.ismethod)
        if not name.startswith("_")
    }

    for name in method_names:
        if name in available_methods:
            func = available_methods[name]
            signature = inspect.signature(func)
            docstring = inspect.getdoc(func) or "No documentation available."
            docs.append(f'def {name}{signature}:\n    """{docstring}"""\n')
        else:
            docs.append(f"Method '{name}' not found in WorldAPI.")

    return "\n\n".join(docs)
