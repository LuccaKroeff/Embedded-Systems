import click
import json
import csv
import sys
from platforms import supported_platforms, BasePlatform
from data import Query
import pathlib


@click.command()
@click.argument("platform", type=click.Choice(supported_platforms, case_sensitive=False))
#@click.argument("queries_file", type=click.File(mode="r", encoding="utf-8"))
#@click.option("--port", default='/dev/ttyUSB0')
#@click.option("--skip-compilation", is_flag=True)
@click.option("--output-file", default=sys.stdout, type=click.File(mode="w", encoding="utf-8"))
def main(platform, output_file):
    platform_class = supported_platforms[platform]
    platform: BasePlatform = platform_class()

    results = []
    print(f"Running benchmarks on platform '{platform.name}'")
    for core in platform.supported_cores:
        print(f"Running core {core.name}")
        if platform.name == 'esp32':
            core.queries_file = 'sql_queries_reduced.json'
        platform.deploy_core(core)
        queries = json.loads(pathlib.Path(core.queries_file).read_text())
        queries = [Query(**document) for document in queries]

        results.extend(platform.run_queries(queries))
        platform.remove_core()

    writer = csv.writer(output_file)
    writer.writerow(["test", "database", "time", "energy", "edp"])
    for result in results:
        writer.writerow(result.to_list())


if __name__ == '__main__':
    main()