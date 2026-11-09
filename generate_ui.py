"""用当前解释器的 pyuic5 生成主窗和零位设定界面。"""

import os
from pathlib import Path
import subprocess
import sys


def generate():
    directory = Path(__file__).resolve().parent
    for name in ("Form1", "ZeroDialog"):
        source = subprocess.check_output(
            [sys.executable, "-m", "PyQt5.uic.pyuic", f"{name}.ui"],
            cwd=directory, encoding="utf-8", env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        )
        header = f"# 本文件由 {name}.ui 自动生成；修改界面后请运行 generate_ui.py。\n\n"
        (directory / f"ui_{name}.py").write_text(
            header + source[source.index("from PyQt5 import"):], encoding="utf-8"
        )


if __name__ == "__main__":
    generate()
