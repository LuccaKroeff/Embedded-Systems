import subprocess
from itertools import product
import pandas as pd
import json
from pprint import pp

progs = [
    'sobel_sequential',
    'sobel_openmp',
    'sobel_simd',
    'sobel_simd_openmp'
]

sizes = [
    512,
    1024,
    2048,
    4096,
    8192,
    16384
]
num_reps = 20

def parse_output(line):
    return float(line[line.find(": ")+2:])

df = pd.DataFrame(columns=['program', 'input_size', 'real_time', 'cpu_time', 'energy', 'cycles', 'instructions'])

for prog, size in product(progs, sizes):
    real_times = []
    cpu_times = []
    energy = []
    print(f"Running {prog} with input size {size}")
    for i in range(num_reps):
        output = subprocess.run(
            ["perf", "stat", "-j", f"./build/{prog}", f"{size}"], 
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, 
            encoding="utf8",
            env={"LC_ALL": "C"}
        )
        _, real_time_line, cpu_time_line, energy_line = output.stdout.splitlines()
        perf_output = [json.loads(a) for a in output.stderr.splitlines()]

        for event in perf_output:
            match event:
                case {'event': 'cycles', 'counter-value': value}:
                    cycles = value
                case {'event': 'instructions', 'counter-value': value}:
                    instructions = value
        

        df.loc[len(df)] = [
            prog,
            size,
            parse_output(real_time_line),
            parse_output(cpu_time_line),
            parse_output(energy_line),
            cycles,
            instructions
        ]

df.to_csv("results.csv")        
