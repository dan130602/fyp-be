"""Replay decoder: answers from a table of saved texts, keyed by (recording id, start sample, end sample). For tests and verification tools only: the command line never selects it.

A key that is not in the table is an error (ReplayMissing), never an empty text. A table with the same key twice and different texts is refused when it is built.
"""
from meralion_worker.decoders.base import DecodeResult, Decoder, validate_piece


class ReplayMissing(KeyError):
    pass


def build_table(entries):
    """entries: iterable of ((recording_id, start_sample, end_sample), DecodeResult | str). Returns a dict of DecodeResult."""
    table = {}
    for key, value in entries:
        result = value if isinstance(value, DecodeResult) else DecodeResult(text=value, tokens=None, stop_reason=None, decode_seconds=0.0)
        key = (key[0], int(key[1]), int(key[2]))
        if key in table and table[key].text != result.text:
            raise ValueError(f"replay table has two different texts for {key}")
        table[key] = result
    return table


class ReplayDecoder(Decoder):
    def __init__(self, table):
        self.table = table if all(isinstance(v, DecodeResult) for v in table.values()) else build_table(table.items())
        self.calls = []                                               # every (recording_id, start, end) asked for, in order (tests inspect this)

    def decode(self, piece, *, recording_id, start_sample, end_sample):
        validate_piece(piece, start_sample, end_sample)
        key = (recording_id, int(start_sample), int(end_sample))
        self.calls.append(key)
        try:
            return self.table[key]
        except KeyError:
            raise ReplayMissing(f"no saved decode for {key}") from None

    def describe(self):
        return {"backend": "replay", "settings_id": "replay", "frozen_digest": "replay", "runtime": {"packages": {}, "patches": []}, "note": "saved texts; no inference"}
