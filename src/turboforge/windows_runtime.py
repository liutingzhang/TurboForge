"""Read PE imports and collect app-local redistributable DLLs for a release."""
from __future__ import annotations
import os
from pathlib import Path
import struct


def imports(path: Path) -> list[str]:
    with path.open("rb") as stream:
        def read(offset, size):
            stream.seek(offset)
            return stream.read(size)
        if read(0, 2) != b"MZ":
            raise ValueError(f"Not a Windows executable: {path.name}")
        pe = struct.unpack("<I", read(60, 4))[0]
        if read(pe, 4) != b"PE\0\0":
            raise ValueError(f"Invalid PE header: {path.name}")
        count = struct.unpack("<H", read(pe + 6, 2))[0]
        size = struct.unpack("<H", read(pe + 20, 2))[0]
        optional = pe + 24
        magic = struct.unpack("<H", read(optional, 2))[0]
        directory = optional + (112 if magic == 0x20b else 96)
        rva = struct.unpack("<I", read(directory + 8, 4))[0]
        if not rva:
            return []
        sections = [struct.unpack("<4I", read(optional + size + i * 40 + 8, 16)) for i in range(count)]
        def offset(value):
            for virtual_size, address, raw_size, pointer in sections:
                if address <= value < address + max(virtual_size, raw_size):
                    return pointer + value - address
            raise ValueError(f"PE address out of bounds: {path.name}")
        position = offset(rva)
        result = []
        while True:
            fields = struct.unpack("<5I", read(position, 20))
            if not any(fields):
                return result
            result.append(read(offset(fields[3]), 256).split(b"\0")[0].decode("ascii"))
            position += 20


def collect(engine: Path) -> dict[str, Path]:
    system = Path(os.environ.get("SystemRoot", "C:/Windows")) / "System32"
    search = [engine, system, *[Path(p) for p in os.environ.get("PATH", "").split(os.pathsep) if p]]
    files = {p.name.lower(): p for p in engine.iterdir() if p.suffix.lower() in (".dll", ".exe")}
    queue = list(files.values())
    while queue:
        path = queue.pop()
        for name in imports(path):
            key = name.lower()
            if key in files or key.startswith(("api-ms-win-", "ext-ms-win-")):
                continue
            dependency = next((directory / name for directory in search if (directory / name).is_file()), None)
            if dependency is None:
                raise RuntimeError(f"The release is missing runtime {name} (referenced by {path.name}); a distributable bundle cannot be produced")
            # Windows OS/driver files stay on the destination OS. Only the
            # Microsoft C++ redistributable DLL family is copied app-locally.
            if dependency.parent == system and not key.startswith(("msvcp", "vcruntime", "concrt")):
                continue
            files[key] = dependency
            queue.append(dependency)
    return files
