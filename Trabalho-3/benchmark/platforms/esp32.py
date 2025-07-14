import subprocess
import pexpect

from cores import SQLiteCore, RocksDBCore
from data import Result

from .base import BasePlatform

class ESP32(BasePlatform):
    name = "esp32"
    supported_cores = [SQLiteCore]
    board_id = "esp32:esp32:esp32-poe-iso"
    tdp = 0.5

    def __init__(self, port="/dev/ttyUSB0"):
        self.port = port
        self.sketch = None
        self.monitor = None
    
    def deploy_core(self, core):
        if core not in self.supported_cores:
            raise ValueError("Core not supported")

        self.compile(core.name)
        self.upload(core.name)

        self.core = core
        self.monitor = pexpect.spawn("arduino-cli", ["monitor", "--port", self.port, "--config", "115200"])

    def remove_core(self):
        self.monitor.close()
        self.core = None

    def compile(self, sketch):
        subprocess.run(
            ["arduino-cli", "compile", "--fqbn", self.board_id, f"./code/sketches/{sketch}"],
            encoding="utf8",
        )
    
    def upload(self, sketch):    
        subprocess.run(
            ['arduino-cli', 'upload', '-p', self.port, '--fqbn', self.board_id, f"./code/sketches/{sketch}"],
            encoding="utf8",
        )
    
    def run_query(self, query):
        cmd = query.command
        iterations = str(query.iterations)

        self.monitor.expect("Q:")
        self.monitor.send(cmd + "\n")
        self.monitor.expect("I:")
        self.monitor.send(iterations + "\n")
        self.monitor.expect("T: ")
        time = float(self.monitor.readline().decode().strip())

        return Result(query=query, core=self.core, time=time, energy=self.tdp*time)