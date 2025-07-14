from dataclasses import dataclass, field
from cores import Core

@dataclass
class Query:
    name: str
    command: str
    iterations: int = 1
    slots: int = 0


@dataclass
class Result:
    query: Query
    core: Core
    time: float
    energy: float
    
    @property
    def edp(self):
        return self.time * self.energy

    def to_list(self):
        return [self.core.name, self.query.name, self.time, self.energy, self.edp]
