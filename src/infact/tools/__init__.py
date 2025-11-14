from infact.common.logger import Logger
from infact.tools.search.searcher import Searcher
from infact.tools.tool import Tool, get_available_actions

TOOL_REGISTRY = [
    Searcher,
]


def get_tool_by_name(name: str):
    for t in TOOL_REGISTRY:
        if t.name == name:
            return t
    raise ValueError(f'Tool with name "{name}" does not exist.')

class FakeLogger:
    def log(self, message):
        pass
    def info(self, message):
        pass
    def warning(self, message):
        pass
    def error(self, message):
        pass
    def debug(self, message):
        pass

def initialize_tools(config: dict[str, dict], logger= None, device=None) -> list[Tool]:
    tools = []
    for tool_name, kwargs in config.items():
        tool_class = get_tool_by_name(tool_name)
        if device is not None:
            kwargs["device"] = device
        if logger is None:
            kwargs["logger"] = FakeLogger()
        else:
            kwargs["logger"] = logger
        t = tool_class(**kwargs)
        tools.append(t)
    return tools
 