import subprocess
import pexpect
import pathlib
from monitors import PexpectMonitor

from .base import BasePlatform

class Linux_x86(BasePlatform):
    name = "Linux x86"
    
    def __init__(self):
        self.app = None
        self.monitor = None
    
    def compile(self, app, cwd=None):
        subprocess.run(["make", app], cwd=cwd)

    def run(self, app, cwd=None):
        if self.app is not None:
            raise ValueError(f"Platform is alrady running app {self.app}")

        self.app = app
        exec_path: pathlib.Path = str(pathlib.Path(cwd) / app)
        self.monitor = PexpectMonitor(pexpect.spawn(exec_path))
        return self.monitor

    def clean(self, cwd=None):
        if self.app is None:
            return
        self.monitor.finish()
        self.monitor = None
        self.app = None
        subprocess.run(["make", "clean"], cwd=cwd)
