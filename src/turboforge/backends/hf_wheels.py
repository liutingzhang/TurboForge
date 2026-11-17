"""Persistent, publisher-hashed Windows CUDA wheel downloads (no pip temp loss)."""
from __future__ import annotations
from html.parser import HTMLParser
from pathlib import Path
import re
import subprocess
from urllib.parse import unquote, urlsplit, urlunsplit

from ..assistant.config import child_environment


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.hrefs = []

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            self.hrefs.extend(value for key,value in attrs if key == "href" and value)


def cuda_wheels(python: Path, cache: Path, emit) -> list[Path]:
    from ..bootstrap import download, read_url
    abi = subprocess.run([str(python),"-c","import sys;print(f'cp{sys.version_info.major}{sys.version_info.minor}')"],
                         capture_output=True,text=True,check=True,env=child_environment()).stdout.strip()
    if abi not in ("cp310","cp313"):
        raise RuntimeError("The current build-free CUDA environment only supports the CPython 3.10 / 3.13 ABI")
    wheels = []
    for name,version in (("torch","2.14.1"),("torchvision","0.29.1")):
        filename = f"{name}-{version}+cu130-{abi}-{abi}-win_amd64.whl"
        parser = Links()
        parser.feed(read_url(f"https://download.pytorch.org/whl/cu130/{name}/").decode())
        candidates = [urlsplit(link) for link in parser.hrefs if unquote(urlsplit(link).path).rsplit("/",1)[-1] == filename]
        if len(candidates) != 1:
            raise RuntimeError("The official CUDA index has no unique matching wheel: "+filename)
        url = candidates[0]
        if url.scheme != "https" or url.netloc not in ("download.pytorch.org","download-r2.pytorch.org") or not re.fullmatch(r"sha256=[0-9a-f]{64}",url.fragment):
            raise RuntimeError("The CUDA wheel source or release digest is invalid; installation refused")
        path = download(urlunsplit(url._replace(fragment="")),cache/filename,sha256=url.fragment[7:],emit=emit)
        wheels.append(path)
    return wheels
