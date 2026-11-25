"""Check driver/worker Python paths in the existing uv or venv environment."""
import argparse
import json
import os
from pathlib import Path
import secrets

from combatsim.ray_runtime import configure_ray_runtime, interpreter_identity, same_interpreter

configure_ray_runtime()  # Must precede the first Ray import.

import ray


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default=str(root/'artifacts/runtime-check/report.json'))
    parser.add_argument('--ray-temp-dir', default=str(root/'artifacts/runtime-check/ray'))
    args = parser.parse_args()
    output, temporary = Path(args.output).absolute(), Path(args.ray_temp_dir).absolute()
    temporary.mkdir(parents=True, exist_ok=True)
    driver = interpreter_identity()
    os.environ.setdefault('RAY_AUTH_TOKEN', secrets.token_urlsafe(32))
    ray.init(address='local', num_cpus=2, include_dashboard=False, _temp_dir=str(temporary))
    try:
        @ray.remote(num_cpus=.5)
        class Probe:
            def identity(self):
                return interpreter_identity()
        actors = [Probe.remote() for _ in range(2)]
        workers = ray.get([actor.identity.remote() for actor in actors], timeout=90)
        ok = all(same_interpreter(driver, worker) for worker in workers)
        report = {'ok': ok, 'driver': driver, 'workers': workers,
                  'automatic_uv_runtime': os.environ['RAY_ENABLE_UV_RUN_RUNTIME_ENV']}
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        print(json.dumps(report, ensure_ascii=False, indent=2))
        if not ok:
            raise RuntimeError('Driver and Ray workers use different interpreters or CombatSim installations')
        print('OK: both Ray workers use the driver Python, virtual environment and CombatSim installation')
    finally:
        ray.shutdown()


if __name__ == '__main__':
    main()
