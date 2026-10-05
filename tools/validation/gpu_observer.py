"""Read-only 50 ms device-memory sampling. No model or RNG interaction."""
from __future__ import annotations
import datetime as dt
import json
import subprocess
import threading
import time
from pathlib import Path


class DeviceMemoryObserver:
    def __init__(self, directory: Path, device: int = 0):
        self.directory = directory
        self.device = device
        self.samples = []
        self.errors = []
        self.ready = threading.Event()

    def __enter__(self):
        self.directory.mkdir(parents=True, exist_ok=True)
        self.started_at = dt.datetime.now(dt.timezone.utc).isoformat()
        self.start = time.perf_counter()
        self.process = subprocess.Popen(
            ["nvidia-smi", "--id", str(self.device), "--query-gpu=timestamp,memory.used",
             "--format=csv,noheader,nounits", "--loop-ms=50"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8")
        self.thread = threading.Thread(target=self._read, daemon=True)
        self.thread.start()
        if not self.ready.wait(5) or not self.samples:
            self.__exit__(None, None, None)
            raise RuntimeError("50 ms device-memory sampling is unavailable")
        return self

    def _read(self):
        with (self.directory / "device-memory.csv").open("w", encoding="utf-8") as log:
            log.write("elapsed_seconds,timestamp,used_mib\n")
            for line in self.process.stdout:
                try:
                    stamp, used = line.strip().rsplit(",", 1)
                    value = float(used.strip())
                    if value < 0 or not value < float("inf"):
                        raise ValueError("invalid memory sample")
                    elapsed = time.perf_counter() - self.start
                    self.samples.append((elapsed, value))
                    log.write(f"{elapsed:.9f},{stamp},{value}\n")
                    log.flush()
                    self.ready.set()
                except ValueError as error:
                    self.errors.append(str(error))

    def __exit__(self, *_):
        if self.process.poll() is None:
            self.process.terminate()
        self.process.wait(timeout=10)
        self.thread.join(timeout=10)
        self.process.stdout.close()
        error = self.process.stderr.read().strip()
        self.process.stderr.close()
        if error:
            self.errors.append(error)
        dropped = sum(max(0, round((b[0] - a[0]) / .05) - 1)
                      for a, b in zip(self.samples, self.samples[1:]))
        result = dict(format_version=1, scope="device-wide sampled used memory; includes other processes",
                      interval_ms=50, device=self.device, started_at=self.started_at,
                      ended_at=dt.datetime.now(dt.timezone.utc).isoformat(),
                      seconds=time.perf_counter()-self.start, samples=len(self.samples),
                      estimated_dropped_samples=dropped, errors=self.errors,
                      baseline_used_mib=self.samples[0][1] if self.samples else None,
                      peak_used_mib=max(x[1] for x in self.samples) if self.samples else None,
                      process_memory="unavailable; device total is not process allocation")
        (self.directory / "device-memory.json").write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with DeviceMemoryObserver(args.output):
        time.sleep(.25)
