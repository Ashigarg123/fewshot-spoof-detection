from pathlib import Path


def read_manifest(manifest_path, audio_root=None):
    manifest_path = Path(manifest_path)
    root = Path(audio_root) if audio_root else manifest_path.parent
    entries = []

    with manifest_path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            fields = line.rsplit(maxsplit=1)
            if len(fields) != 2:
                raise ValueError(
                    f'{manifest_path}:{line_number}: expected "audio_path label"'
                )
            audio_path, label = fields
            audio_path = Path(audio_path)
            if not audio_path.is_absolute():
                audio_path = root / audio_path
            entries.append((str(audio_path), label))

    if not entries:
        raise ValueError(f'{manifest_path}: manifest is empty')
    return entries
