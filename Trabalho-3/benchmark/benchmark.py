from data import Result, Input
import json
import pathlib
import csv


class BenchmarkRun:
    def __init__(self, platform, target, app, input_file, cwd):
        self.platform = platform
        self.target = target
        self.app: str = app
        json_inputs = json.loads(pathlib.Path(input_file).read_text())
        self.inputs = [Input(inp["name"], inp["inputs"]) for inp in json_inputs]
        self.cwd: str = cwd

    def run(self) -> list[Result]:
        self.compile_code()
        monitor = self.deploy_code()
        results = []
        for input_case in self.inputs:
            print(f"Running '{input_case.name}'")

            for input_line in input_case.inputs:
                monitor.send_input(input_line["input_marker"], input_line["input_value"])

            result = monitor.read_response()
            result_obj = Result(
                input=input_case, 
                platform=self.platform.name, 
                app=self.app, 
                output=result
            )
            results.append(result_obj)
        self.clean()
        return results

    def compile_code(self):
        print(f"Compiling {self.app} for {self.platform.name}")
        self.platform.compile(self.target, cwd=self.cwd)
    
    def deploy_code(self):
        print(f"Deploying {self.app} for {self.platform.name}")
        return self.platform.run(self.target, cwd=self.cwd)
    
    def clean(self):
        return self.platform.clean(cwd=self.cwd)


class BaseBenchmark:
    def run(self):
        results = []
        for test in self.tests:
            run = BenchmarkRun(
                platform=test["platform"](), 
                app=test["app"],
                target=test["target"],
                input_file=test["input_file"],
                cwd=test["cwd"]
            )
            results.extend(run.run())
        
        with open(self.output_file, "w") as output_file:
            writer = csv.writer(output_file)
            writer.writerow(["app", "platform", "test"] + self.output_columns)
            
            for result in results:
                writer.writerow([
                    result.app,
                    result.platform,
                    result.input.name,
                    *result.output.split(",")
                ])
