import importlib.metadata

try:
    __version__ = importlib.metadata.version("vdaworld")
except importlib.metadata.PackageNotFoundError:
    # Package is not installed
    __version__ = "unknown"


def hello() -> str:
    return "Hello from vdaworld!"
