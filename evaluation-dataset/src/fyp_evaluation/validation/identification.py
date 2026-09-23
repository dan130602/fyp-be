"""Synthetic identifier checks, not identity verification or a registry lookup.

Weights, offsets and alphabets checked against:
https://github.com/IonBazan/NRIC/blob/main/src/NRIC.php
ST/FG rules also cross-checked against:
https://gist.github.com/alesmana/db3d215d649f987ff30e
The pilot uses only S, F and G prefixes.
"""

import re


def is_invalid_test_identifier(value):
    """Require the requested format and a deliberately mismatched checksum."""
    if not isinstance(value, str) or not re.fullmatch(r"[STFGM][0-9]{7}[A-Z]", value):
        return False
    prefix = value[0]
    alphabet = ("JZIHGFEDCBA" if prefix in "ST" else
                "XWUTRQPNJLK" if prefix == "M" else "XWUTRQPNMLK")
    offset = 3 if prefix == "M" else 4 if prefix in "TG" else 0
    total = offset + sum(int(digit) * weight for digit, weight in
                         zip(value[1:8], (2, 7, 6, 5, 4, 3, 2)))
    return value[-1] != alphabet[total % 11]
