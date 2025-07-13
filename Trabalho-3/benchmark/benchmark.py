import click
import json
import csv
import sys
from platforms import supported_platforms, BasePlatform
from data import Query



# board = "esp32:esp32:esp32-poe-iso"

# def compile_sketch(sketch):
#     print(f"Compiling program: {sketch}")
#     result = subprocess.run(
#         ["arduino-cli", "compile", "--fqbn", board, f"./sketches/{sketch}"],
#         encoding="utf8",
#     )
#     if result.returncode != 0:
#         print("Error on sketch compilation")
#         exit()


# def upload_sketch(sketch, port):
#     print(f"Uploading sketch '{sketch}' to board at port '{port}'")
#     subprocess.run(
#         ['arduino-cli', 'upload', '-p', port, '--fqbn', board, f"./sketches/{sketch}"],
#         encoding="utf8",
#     )

# def send_queries(sketch, port, queries):
#     proc = pexpect.spawn(
#         "arduino-cli", ["monitor", "--port", port, "--config", "115200"]
#     )
#     results = []
#     for query in queries:
#         name = query.get("name")
#         print(f"Running test {name}")

#         cmd = query.get("query")
#         iterations = str(query.get("iterations", 1))


#         proc.expect("Q:")
#         proc.send(cmd + "\n")
#         proc.expect("I:")
#         proc.send(iterations + "\n")
#         proc.expect("T: ")
#         result = float(proc.readline().decode().strip())

#         results.append((sketch, name, result))

#     return results


@click.command()
@click.argument("platform", type=click.Choice(supported_platforms, case_sensitive=False))
@click.argument("queries_file", type=click.File(mode="r", encoding="utf-8"))
#@click.option("--port", default='/dev/ttyUSB0')
#@click.option("--skip-compilation", is_flag=True)
@click.option("--output-file", default=sys.stdout, type=click.File(mode="w", encoding="utf-8"))
def main(platform, queries_file, output_file):
    queries = json.load(queries_file)
    queries = [Query(**document) for document in queries]

    platform_class = supported_platforms[platform]
    platform: BasePlatform = platform_class()

    print(f"Running benchmarks on platform {platform}")
    for core in platform.supported_cores:
        print(f"Running core {core.name}")
        platform.deploy_core(core)
        results = platform.run_queries(queries)
        platform.remove_core()

    writer = csv.writer(output_file)
    writer.writerow(["test", "database", "time", "energy", "edp"])
    for result in results:
        writer.writerow(result.to_list())


if __name__ == '__main__':
    main()