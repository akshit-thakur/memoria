"""Stop supervisord when a required child exhausts its restart attempts."""

from __future__ import annotations

import os
import signal
import sys


def main() -> None:
    while True:
        sys.stdout.write("READY\n")
        sys.stdout.flush()

        header_line = sys.stdin.readline()
        if not header_line:
            return
        headers = dict(item.split(":", 1) for item in header_line.split())
        payload = sys.stdin.read(int(headers["len"]))

        print(
            f"required process entered FATAL state: {payload.strip()}",
            file=sys.stderr,
            flush=True,
        )
        sys.stdout.write("RESULT 2\nOK")
        sys.stdout.flush()
        os.kill(os.getppid(), signal.SIGTERM)
        return


if __name__ == "__main__":
    main()

