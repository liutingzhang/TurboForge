import os
import subprocess
import sys
import pytest
from turboforge.process_owner import ServerJob


@pytest.mark.skipif(os.name != "nt",reason="Windows job object")
def test_model_child_is_released_when_job_closes():
    owner=ServerJob()
    child=subprocess.Popen([sys.executable,"-c","import time;time.sleep(60)"])
    try:
        owner.attach(child.pid)
        owner.close()
        assert child.wait(timeout=5) is not None
    finally:
        owner.close()
        if child.poll() is None:child.terminate();child.wait(timeout=5)
