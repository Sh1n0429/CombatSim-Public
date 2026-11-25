"""Check Ray's uv bootstrap without starting a cluster."""
import os
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch
from combatsim.ray_runtime import configure_ray_runtime, same_interpreter

ROOT=Path(__file__).resolve().parents[1]

class RuntimeChecks(unittest.TestCase):
    def test_bootstrap_precedes_ray_import_and_blocks_uv_relaunch(self):
        environment=os.environ.copy();environment.pop('RAY_ENABLE_UV_RUN_RUNTIME_ENV',None)
        code='''
import scripts.train
from ray._private import ray_constants, worker
from unittest.mock import patch
assert not ray_constants.RAY_ENABLE_UV_RUN_RUNTIME_ENV
with patch('ray._private.runtime_env.uv_runtime_env_hook._get_uv_run_cmdline', return_value=['uv','run','python','scripts/train.py']), patch('ray._private.runtime_env.uv_runtime_env_hook.hook', side_effect=AssertionError('uv must not relaunch')):
    assert worker._maybe_modify_runtime_env(None, False) is None
'''
        result=subprocess.run([sys.executable,'-c',code],cwd=ROOT,env=environment,
                              capture_output=True,text=True,timeout=45)
        self.assertEqual(result.returncode,0,result.stderr)
    def test_explicit_uv_runtime_opt_in_is_preserved(self):
        with patch.dict(os.environ,{'RAY_ENABLE_UV_RUN_RUNTIME_ENV':'1'}):
            configure_ray_runtime()
            self.assertEqual(os.environ['RAY_ENABLE_UV_RUN_RUNTIME_ENV'],'1')
    def test_base_interpreter_alone_does_not_identify_a_venv(self):
        driver={'executable':str(ROOT/'.venv/bin/python'), 'prefix':str(ROOT/'.venv'), 'combatsim':str(ROOT/'combatsim/__init__.py')}
        self.assertTrue(same_interpreter(driver,dict(driver)))
        self.assertFalse(same_interpreter(driver,{**driver,'prefix':str(ROOT/'other-venv')}))
        self.assertFalse(same_interpreter(driver,{**driver,'executable':str(ROOT/'base/bin/python')}))

if __name__=='__main__':
    unittest.main(verbosity=2)
