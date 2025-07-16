from data import Result
from abc import ABC, abstractmethod

class BasePlatform(ABC):
    supported_cores = []

    def run_queries(self, queries) -> list[Result]:
        results = []
        for query in queries:
            print(f"Query: {query.name}")
            result = self.run_query(query)
            results.append(result)
        return results