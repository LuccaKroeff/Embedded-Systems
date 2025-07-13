import subprocess
import pexpect
from .base import BasePlatform
from cores import SQLiteCore, RocksDBCore, DuckDBCore
from data import Result

class Linux_x86(BasePlatform):
    supported_cores = [SQLiteCore, RocksDBCore, DuckDBCore]
    
    def __init__(self):
        self.process = None
    
    def deploy_core(self, core):
        self.compile(core)
        self.process = pexpect.spawn(f"./{core.name}")

    def compile(self, core):
        subprocess.run("gcc", ["-O2", f"{core.name}.c", "-o", core.name])

    def run_query(self, query):
        cmd = query.command
        iterations = str(query.iterations)

        self.process.expect("Q:")
        self.process.send(cmd + "\n")
        self.process.expect("I:")
        self.process.send(iterations + "\n")
        self.process.expect("T: ")
        time = float(self.process.readline().decode().strip())
        self.process.expect("E: ")
        energy = float(self.process.readline().decode().strip())

        return Result(query=query, core=self.core, time=time, energy=energy)
    
    def remove_core(self, core):
        self.process.kill()
        self.process.wait()
        self.core = None