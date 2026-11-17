from types import SimpleNamespace

import pytest

from turboforge.backends.hf_wheels import cuda_wheels


def test_cuda_wheels_use_publisher_hashes_and_persistent_downloads(tmp_path,monkeypatch):
    monkeypatch.setattr('turboforge.backends.hf_wheels.subprocess.run',lambda *a,**kw:SimpleNamespace(stdout='cp313'))
    def page(url):
        name='torchvision' if '/torchvision/' in url else 'torch'
        version='0.29.1' if name=='torchvision' else '2.14.1'
        return f'<a href="https://download-r2.pytorch.org/whl/cu130/{name}-{version}%2Bcu130-cp313-cp313-win_amd64.whl#sha256={"a"*64}">wheel</a>'.encode()
    monkeypatch.setattr('turboforge.bootstrap.read_url',page)
    calls=[]
    def download(url,destination,**kwargs):
        calls.append((url,destination,kwargs));return destination
    monkeypatch.setattr('turboforge.bootstrap.download',download)
    results=cuda_wheels(tmp_path/'python.exe',tmp_path/'persistent',lambda _:None)
    assert len(results)==2 and all(path.parent==tmp_path/'persistent' for path in results)
    assert all(args['sha256']=='a'*64 for _,_,args in calls)
    assert all('#' not in url for url,_,_ in calls)


def test_untrusted_wheel_host_never_downloaded(tmp_path,monkeypatch):
    monkeypatch.setattr('turboforge.backends.hf_wheels.subprocess.run',lambda *a,**kw:SimpleNamespace(stdout='cp313'))
    monkeypatch.setattr('turboforge.bootstrap.read_url',lambda _:f'<a href="https://evil.example/torch-2.14.1%2Bcu130-cp313-cp313-win_amd64.whl#sha256={"a"*64}">x</a>'.encode())
    with pytest.raises(RuntimeError,match='source or release digest'):
        cuda_wheels(tmp_path/'python.exe',tmp_path,lambda _:None)
