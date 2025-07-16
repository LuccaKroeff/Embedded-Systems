from .esp32 import ESP32
from .linux_x86 import Linux_x86
from .android import Android
from .base import BasePlatform

supported_platforms = {
    "esp32": ESP32,
    "linux_x86": Linux_x86,
    "android": Android,
}
