import re

HALF_PERCENT = re.compile(r"\bhalf\s+a\s+per\s*cent\b", re.I)
FRACTION_PREFIX = re.compile(
    r"(?:\w+[-–]|\b(?:and|one|two|three|four|five|six|seven|eight|nine|[0-9]+)\s+(?:a\s+)?)$",
    re.I,
)

def format_fee(text):
    """Render standalone half-percent phrases equivalently; leave compound fractions alone."""
    def replace(match):
        before=text[max(0,match.start()-40):match.start()]
        return match.group() if FRACTION_PREFIX.search(before) else '0.5 percent'
    return HALF_PERCENT.sub(replace,text)
