"""Verified Windows dependency wheels and a compiler-free interpreter choice."""
import os
from pathlib import Path
import subprocess
import sys
from ..assistant.config import child_environment

WHEELS = {
    (3,10): ('pypcre-0.6.4-cp310-cp310-win_amd64.whl','39995fce275dd809763dca4f95edb57d56b0be3968c671952ac1b270135f186f'),
    (3,13): ('pypcre-0.6.4-cp313-cp313-win_amd64.whl','cc6cc732891e65963d4ee5f37e4e82d4f31daed74b8326fb9e0e3922164f27fc'),
}


def bootstrap_python(root: Path, emit) -> Path:
    if os.name != 'nt' or (3,13,18) <= sys.version_info < (3,14):
        return Path(sys.executable)
    from ..bootstrap import run
    target=Path(os.environ['LOCALAPPDATA'])/'NInferTurboForge/Python31318'
    python=target/'python.exe'
    if not python.is_file():
        emit('为免编译依赖自动准备带安全补丁的 Python 3.13.18；不修改系统 PATH')
        script=Path(__file__).parents[1]/'data/install_hf_python.ps1'
        run(['powershell','-NoProfile','-ExecutionPolicy','Bypass','-File',str(script),str(Path(__file__).parents[3]),str(root/'downloads')],emit)
    probe=subprocess.run([str(python),'-c','import sys,struct; assert (3,13,18) <= sys.version_info < (3,14); assert struct.calcsize("P") == 8'],capture_output=True,text=True,env=child_environment())
    if probe.returncode: raise RuntimeError('独立 Python 3.13 验证失败：'+probe.stderr[-1000:])
    return python


def install_pcre(python: Path, emit) -> None:
    if os.name != 'nt': return
    from ..bootstrap import run
    from ..manifest import file_sha256
    result=subprocess.run([str(python),'-c','import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")'],capture_output=True,text=True,check=True,env=child_environment())
    version=tuple(map(int,result.stdout.strip().split('.')))
    if version not in WHEELS:
        raise RuntimeError('已有 HF 环境的 Python 版本缺少免编译依赖，请使用新的 TURBOFORGE_HOME 重新准备')
    name,digest=WHEELS[version]
    wheel=Path(__file__).parents[1]/'data/hf-wheels'/name
    if not wheel.is_file() or file_sha256(wheel) != digest:
        raise RuntimeError('内置 PyPcre wheel 缺失或校验失败，拒绝执行；请重新解压完整发布包')
    run([str(python),'-m','pip','install','--no-deps',str(wheel)],emit)
