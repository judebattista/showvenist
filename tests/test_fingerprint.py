from showvenist.state.fingerprint import content_key, fingerprint


def _write(p, data):
    p.write_bytes(data)
    return p


def test_fingerprint_format_and_header_sensitivity(tmp_path):
    body = bytes(range(256)) * 2048  # 512 KB
    a = _write(tmp_path / "a.bin", body)
    b = _write(tmp_path / "b.bin", b"X" + body[1:])
    c = _write(tmp_path / "c.bin", body)
    assert fingerprint(a).startswith(f"{len(body)}:")
    assert fingerprint(a) == fingerprint(c)
    assert fingerprint(a) != fingerprint(b)


def test_content_key_ignores_header_and_tail(tmp_path):
    body = bytes(i * 7 % 251 for i in range(600_000))
    a = _write(tmp_path / "a.bin", body)
    b = _write(tmp_path / "b.bin", b"HEADER" + body[6:-6] + b"TAILER")
    assert fingerprint(a) != fingerprint(b)
    assert content_key(a) == content_key(b)
    d = bytearray(body)
    d[len(body) // 2 + 10] ^= 0xFF
    assert content_key(_write(tmp_path / "d.bin", bytes(d))) != content_key(a)


def test_small_files(tmp_path):
    for n in (0, 1, 100, 65536, 100_000, 131072):
        p = _write(tmp_path / f"s{n}", b"z" * n)
        assert fingerprint(p).startswith(f"{n}:")
        assert content_key(p).startswith(f"{n}:")
    x = _write(tmp_path / "x", b"a" * 50_000)
    y = _write(tmp_path / "y", b"a" * 49_999 + b"b")
    assert fingerprint(x) != fingerprint(y)
    assert content_key(x) != content_key(y)
