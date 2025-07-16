import subprocess
import pexpect
import pathlib

from .base import BasePlatform
from monitors import PexpectMonitor

class ESP32(BasePlatform):
    name = "esp32"
    board_id = "esp32:esp32:esp32-poe-iso"
    tdp = 0.5

    def __init__(self, port="/dev/ttyUSB0"):
        self.port = port
        self.app = None
        self.monitor = None
    
    def compile(self, app, cwd=None):
        subprocess.run(
            ["arduino-cli", "compile", "--fqbn", self.board_id, app],
            encoding="utf8",
            cwd=cwd
        )

    def run(self, app, cwd=None):
        if self.app is not None:
            raise ValueError(f"Platform is alrady running app {self.app}")

        # Compilation
        app_path = str(pathlib.Path(cwd) / app)
        subprocess.run(
            ['arduino-cli', 'upload', '-p', self.port, '--fqbn', self.board_id, app_path],
            encoding="utf8",
        )

        # Running
        self.app = app
        self.monitor = PexpectMonitor(pexpect.spawn("arduino-cli", ["monitor", "--port", self.port, "--config", "115200"]))
        return self.monitor

    def clean(self, cwd=None):
        self.monitor.finish()
        self.app = None
        self.monitor = None
