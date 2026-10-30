import gzip
from datetime import date

from tse_dzhynsa_bot.collect import write_day


def test_write_day_is_idempotent(tmp_path):
    rows = [{"id": 2, "text": "б"}, {"id": 1, "text": "а"}]
    p = write_day(tmp_path, "ch", date(2026, 1, 1), rows)
    first = p.read_bytes()
    write_day(tmp_path, "ch", date(2026, 1, 1), list(reversed(rows)))
    assert p.read_bytes() == first
    lines = gzip.decompress(first).decode().splitlines()
    assert '"id": 1' in lines[0]
