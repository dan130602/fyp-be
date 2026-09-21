"""Validate one canonical English DOB form; do not normalize predictions."""

from datetime import date

ENGLISH_MONTHS = (
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
)


def is_canonical_dob(value):
    """Accept only calendar-valid `DD Month YYYY` strings, independent of locale."""
    if not isinstance(value, str):
        return False
    parts = value.split(" ")
    if len(parts) != 3:
        return False
    day, month, year = parts
    if (len(day) != 2 or len(year) != 4 or not day.isascii() or not year.isascii()
            or not day.isdigit() or not year.isdigit() or month not in ENGLISH_MONTHS):
        return False
    try:
        date(int(year), ENGLISH_MONTHS.index(month) + 1, int(day))
    except ValueError:
        return False
    return True
