from data import Result
from abc import ABC, abstractmethod

class BasePlatform(ABC):
    supported_cores = []

    @abstractmethod
    def deploy_core(self, core):
        raise NotImplementedError

    @abstractmethod
    def remove_core(self):
        raise NotImplementedError

    def run_queries(self, queries) -> list[Result]:
        results = []
        for query in queries:
            print(f"Query: {query.name}")
            result = self.run_query(query)
            results.append(result)
        return results