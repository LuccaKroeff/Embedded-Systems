import subprocess
import pexpect
import pathlib
from monitors import PexpectMonitor

from .base import BasePlatform

class Android(BasePlatform):
    name = "Android"
    
    def __init__(self):
        self.app = None
        self.monitor = None
    
    def compile(self, app, cwd=None):
        subprocess.run(["make", app], cwd=cwd)

    def run(self, app, cwd=None):
        if self.app is not None:
            raise ValueError(f"Platform is alrady running app {self.app}")

        self.app = app
        subprocess.run(["adb", "push", app, "/tmp/"], cwd=cwd)
        subprocess.run(["adb", "shell", f'chmod +x /tmp/{app}'])
        proc = pexpect.spawn("adb", ["shell"])
        proc.send(f"/tmp/{app}\n")
        self.monitor = PexpectMonitor(proc)
        return self.monitor

    def clean(self, cwd=None):
        if self.app is None:
            return
        self.monitor.finish()
        self.monitor = None
        self.app = None
        subprocess.run(["make", "clean"], cwd=cwd)
