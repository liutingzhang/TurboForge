"""Exercise a fresh Web environment with developer tools removed from PATH.

This shares the host OS/driver and any managed interpreter. It is not a clean VM
test, does not quantize a model and never copies credentials into the test tree.
"""
from __future__ import annotations
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import urllib.parse
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
from turboforge.assistant.config import child_environment


def check(project: Path) -> dict:
    project = project.resolve()
    if os.name != "nt" or not (project/"scripts/start_web.ps1").is_file():
        raise RuntimeError("An extracted Windows release project is required")
    if (project/".env").exists() or (project/".turboforge/venv-py31318").exists():
        raise RuntimeError("The test directory must contain no personal configuration or existing web environment")
    environment = {key:value for key,value in os.environ.items()
                   if not key.upper().startswith(("PYTHON", "TURBOFORGE_", "VIRTUAL_ENV", "NODE_"))}
    system = Path(os.environ["SystemRoot"])/"System32"
    environment["PATH"] = str(system)+os.pathsep+str(system/"WindowsPowerShell/v1.0")
    environment["TURBOFORGE_NO_BROWSER"] = "1"
    missing = {name:shutil.which(name,path=environment["PATH"]) is None for name in ("python", "node", "npm", "git")}
    if not all(missing.values()):
        raise RuntimeError("Development tools unexpectedly present in the stripped PATH")
    log = project.parent/"clean-start.log"
    report = {"developer_tools_absent":missing,"independent_clean_os":False,
              "note":"Fresh isolated Web environment on the existing Windows/driver; no GPU conversion rerun."}
    with log.open("w",encoding="utf-8") as out:
        process = subprocess.Popen([str(system/"WindowsPowerShell/v1.0/powershell.exe"),"-NoProfile","-ExecutionPolicy","Bypass",
                                    "-File",str(project/"scripts/start_web.ps1")],cwd=project,
                                   env=child_environment(environment),stdout=out,stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic()+240
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError(f"Launcher exited with {process.returncode}; check {log}")
                text=log.read_text(encoding="utf-8",errors="replace")
                match=re.search(r"TurboForge: (http://127\.0\.0\.1:\d+)",text)
                if match:
                    url=match.group(1)
                    try:
                        with urllib.request.urlopen(url,timeout=2) as response:
                            report["page_status"]=response.status
                        with urllib.request.urlopen(url+"/api/assistant/help?q="+urllib.parse.quote("Can I still use MTP after quantization"),timeout=2) as response:
                            report["help_first"]=json.load(response)["entries"][0]["id"]
                        break
                    except OSError:
                        pass
                time.sleep(.5)
            else:
                raise RuntimeError(f"First start exceeded 240 seconds; check {log}")
            python=project/".turboforge/venv-py31318/Scripts/python.exe"
            result=subprocess.run([str(python),"-c","import sys,site,importlib.util,json; print(json.dumps({'isolated':sys.prefix!=sys.base_prefix and not site.ENABLE_USER_SITE,'torch_absent':importlib.util.find_spec('torch') is None}))"],
                                  capture_output=True,text=True,check=True,env=child_environment(environment))
            report.update(json.loads(result.stdout))
            subprocess.run([str(python),"-m","pip","check"],capture_output=True,check=True,env=child_environment(environment))
            report["dependency_check"]="passed"
            report["python_runtime_version"]=subprocess.run([str(python),"-c","import sys;print('.'.join(map(str,sys.version_info[:3])))"],capture_output=True,text=True,check=True,env=child_environment(environment)).stdout.strip()
            report["python_runtime_prepared"]=report["python_runtime_version"] == "3.13.18"
            report["node_installed"]=(project/".turboforge/node").exists()
            if report["page_status"] != 200 or report["help_first"] != "F03" or not report["isolated"] or not report["torch_absent"] or report["node_installed"]:
                raise RuntimeError("First-start check did not meet expectations: "+json.dumps(report))
            return report
        finally:
            if process.poll() is None:
                subprocess.run([str(system/"taskkill.exe"),"/PID",str(process.pid),"/T","/F"],capture_output=True,env=child_environment(environment))
                process.wait(timeout=15)


if __name__ == "__main__":
    result=check(Path(sys.argv[1]))
    (ROOT/"reports").mkdir(exist_ok=True)
    (ROOT/"reports/CLEAN_START_ACCEPTANCE.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(result,ensure_ascii=False,indent=2))
