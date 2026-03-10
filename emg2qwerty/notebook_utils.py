from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path


def run_and_print(
    cmd: list[str],
    cwd: Path,
    log_file: Path | None = None,
) -> None:
    """Run a command as a child process and stream stdout/stderr line-by-line.

    - Avoids capture_output=True RAM blow-ups on long runs.
    - Optionally tees output to a log file.
    - Forces unbuffered output for real-time visibility.
    """

    env = os.environ.copy()
    env["HYDRA_FULL_ERROR"] = "1"
    env["PYTHONUNBUFFERED"] = "1"

    print(">>", " ".join(cmd))
    if log_file is None:
        stamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        log_file = cwd / "logs" / "notebook_runs" / f"train_{stamp}.log"

    log_file = Path(log_file)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    print("Log file:", log_file)

    p = subprocess.Popen(
        cmd,
        cwd=str(cwd),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    assert p.stdout is not None
    with open(log_file, "w", encoding="utf-8", newline="\n") as log_fp:
        try:
            for line in iter(p.stdout.readline, ""):
                print(line, end="")
                log_fp.write(line)
                log_fp.flush()
        finally:
            p.stdout.close()

    rc = p.wait()
    if rc != 0:
        raise RuntimeError(f"Command failed with exit code {rc}. Full log: {log_file}")

