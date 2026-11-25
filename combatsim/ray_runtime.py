"""Bootstrap Ray before import and inspect the actual worker interpreter."""
import os
import sys


def configure_ray_runtime():
    # Dependencies are already installed by uv sync/pip. Ray's automatic uv
    # hook would package the project and launch uv again from a worker copy,
    # potentially selecting/creating another .venv. The local Ray default uses
    # the driver's sys.executable directly. An explicit user opt-in is preserved.
    os.environ.setdefault('RAY_ENABLE_UV_RUN_RUNTIME_ENV', '0')


def interpreter_identity():
    import combatsim
    import ray
    import torch
    return {'pid': os.getpid(), 'executable': sys.executable, 'prefix': sys.prefix,
            'base_prefix': sys.base_prefix, 'cwd': os.getcwd(),
            'combatsim': combatsim.__file__, 'ray_version': ray.__version__,
            'torch_version': torch.__version__}


def same_interpreter(driver, worker):
    # Do NOT resolve symlinks: on Linux .venv/bin/python often links to the
    # base interpreter. Resolving it would erase the virtual environment path.
    def normalize(path):
        return os.path.normcase(os.path.abspath(path))
    return all(normalize(driver[key]) == normalize(worker[key])
               for key in ('executable', 'prefix', 'combatsim'))
