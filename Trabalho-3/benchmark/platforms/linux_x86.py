import subprocess
import pexpect
import pathlib


from .base import BasePlatform
from cores import SQLiteCore, RocksDBCore, DuckDBCore
from data import Result

class Linux_x86(BasePlatform):
    name = "Linux x86"
    supported_cores = [SQLiteCore, DuckDBCore, RocksDBCore]
    
    def __init__(self):
        self.process = None
        self.core = None
    
    def deploy_core(self, core):
        self.compile(core)
        self.core = core
        self.process = pexpect.spawn(f"./code/linux_x86/{core.name}_bench")

    def compile(self, core):
        if core is RocksDBCore:
            cmd = ["g++", "-std=c++17", "-o", "rocksdb_bench", "rocksdb.cpp", "libs/energia.c", "-lrocksdb", "-lz", "-lbz2", "-lsnappy", "-llz4", "-lzstd", "-pthread", '-Wno-unused-result', '-Wno-format-overflow']
        else:
            cmd = ["gcc", "-O2", "-Ilibs", f"{core.name}.c", "libs/energia.c", "-o", f"{core.name}_bench", "-Llibs", "-Wl,-rpath=code/linux_x86/libs",  "-lsqlite3", "-lduckdb", '-Wno-unused-result']
        print(*cmd)
        subprocess.run(cmd, cwd=f"./code/linux_x86")

    def run_query(self, query):
        cmd = query.command
        iterations = str(query.iterations)
        slots = str(query.slots)

        self.process.expect("Q:")
        self.process.send(cmd + "\n")
        self.process.expect("I:")
        self.process.send(iterations + "\n")
        self.process.expect("S:")
        self.process.send(slots + "\n")
        self.process.expect("T: ")
        time = float(self.process.readline().decode().strip())
        self.process.expect("En: ")
        energy = float(self.process.readline().decode().strip())

        return Result(query=query, core=self.core, time=time, energy=energy)

    def remove_core(self):
        self.process.close()
        pathlib.Path(f"code/linux_x86/{self.core.name}_bench").unlink()
        self.core = None