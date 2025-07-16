from dataclasses import dataclass

@dataclass
class Input:
    name: str
    inputs: list[dict]

@dataclass
class Result:
    input: Input
    platform: str
    app: str
    output: str
