"""Schema-2.0 surface catalog: which forms exist for a value, how each one is rendered, and how any surface decodes.

Built once per release from `normalization.json` (schema 2.0). It reads exactly the declarations that already exist:
`spoken_forms` (form ID, language, usage), `generation_surfaces` (the allowlist Python renders from), `aliases` (the
decoding vocabulary), `language_names` (approved product aliases) and `language_rules` (implemented in `zh_numbers`).

Rendering is deterministic: when a form lists several approved surfaces, one is chosen from a hash of the call, question,
value and form, so a value reads the same way throughout a call and a re-run reproduces it. Decoding never receives the
expected value; it maps a surface to a canonical value or raises `RenderError`.

The schema-1.0 (pilot) path does not use this module.
"""

import re
from decimal import Decimal, InvalidOperation
from hashlib import sha256

from . import render as R
from . import zh_numbers as ZH

PROFILE_OF = {
    "K01": "synthetic_nric_fin", "K02": "dob_dd_month_yyyy", "T01": "synthetic_product_name",
    "T02": "currency_iso_4217_alpha3", "CURRENCY": "currency_iso_4217_alpha3", "T03": "monetary_amount",
    "T04": "integer_quantity", "T05": "transaction_direction", "T06": "order_type",
    "FEE": "monetary_amount", "LIMIT_PRICE": "monetary_amount",
}
MONEY_PLACES = 2  # fee amounts and limit prices carry at most two decimal places
RATE_PLACES = 4  # fee rates (percent) carry at most four
_HAN = re.compile(r"[一-鿿〇]")
_EN_NUMBER = re.compile(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?")


def has_han(text):
    return bool(_HAN.search(text))


def _decimal(value):
    try:
        number = value if isinstance(value, Decimal) else Decimal(str(value))
    except InvalidOperation as error:
        raise R.RenderError(f"{value!r} is not a number") from error
    if not number.is_finite() or number < 0:
        raise R.RenderError(f"{value!r} is not a non-negative number")
    return number


def english_decimal(value, max_places):
    """Exact decimal text with comma grouping ('1,500', '37.5'); more decimal places than `max_places` is an error."""
    text = format(_decimal(value).normalize(), "f")
    whole, _, fraction = text.partition(".")
    if len(fraction) > max_places:
        raise R.RenderError(f"{value!r} needs {len(fraction)} decimal places; at most {max_places} can be rendered exactly")
    return f"{int(whole):,}" + (f".{fraction}" if fraction else "")


def parse_english_decimal(text):
    if not _EN_NUMBER.fullmatch(text):
        raise R.RenderError(f"cannot decode number {text!r}")
    return Decimal(text.replace(",", ""))


class SurfaceCatalog:
    def __init__(self, normalization):
        self.profiles = normalization["profiles"]
        self.forms = {(name, form["id"]): form for name, profile in self.profiles.items() for form in profile["spoken_forms"]}
        self._maps = {}

    # ---- forms ---------------------------------------------------------------------------------------
    def form_language(self, name, form):
        return self.forms[(PROFILE_OF[name], form)]["language"]

    def default_forms(self, name):
        """The DEFAULT-usage form IDs of the profile behind a placeholder, in declaration order."""
        profile = PROFILE_OF[name]
        return [form["id"] for (p, _), form in self.forms.items() if p == profile and form["usage"] == "DEFAULT"]

    def surface_options(self, name, value, form):
        """The approved generation surfaces of (value, form), or None when the form renders algorithmically."""
        profile = self.profiles[PROFILE_OF[name]]
        options = (profile.get("generation_surfaces") or {}).get(str(value), {}).get(form)
        return list(options) if options else None

    def form_available(self, name, value, form):
        """Whether `form` can express this particular canonical value (EUR and JPY have no SPELLED_CODE, ...)."""
        profile = PROFILE_OF[name]
        if (profile, form) not in self.forms:
            return False
        body = self.profiles[profile]
        if "generation_surfaces" in body:
            return self.surface_options(name, value, form) is not None
        if name == "T01" and self.form_language(name, form) == "zh":
            entry = (body.get("language_names") or {}).get(str(value), {}).get("zh", {})
            return entry.get("status") == "APPROVED" and bool(entry.get("surface"))
        return True

    @staticmethod
    def _pick(call_id, name, value, form, options):
        key = f"{call_id}|{name}|{value}|{form}".encode("utf-8")
        return options[int.from_bytes(sha256(key).digest()[:8], "big") % len(options)]

    # ---- rendering -----------------------------------------------------------------------------------
    def render_question(self, call_id, name, value, form, detail=None):
        language = self.form_language(name, form)
        options = self.surface_options(name, value, form)
        if options:  # currency, direction and order type render from their approved surfaces in both languages
            return self._pick(call_id, name, value, form, options)
        if language == "zh":
            if name == "K01":
                return ZH.render_identifier(value)
            if name == "K02":
                return ZH.render_date(value)
            if name == "T01":
                entry = (self.profiles["synthetic_product_name"].get("language_names") or {}).get(str(value), {}).get("zh", {})
                if entry.get("status") != "APPROVED":
                    raise R.RenderError(f"no approved Chinese name for {value!r}")
                return entry["surface"]
            if name in ("T03", "T04"):
                if isinstance(value, float) and not value.is_integer() or isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise R.RenderError(f"{value!r} is not a whole number")
                return ZH.render_cardinal(int(value))
            raise R.RenderError(f"unsupported {name} form {form}")
        if name == "K01":
            return R.render_nric(value, form, detail)
        if name == "K02":
            return R.render_dob(value, form)
        if name == "T01":
            return R.render_product(value, form)
        if name in ("T03", "T04"):
            return R.render_amount(value, form, detail)
        raise R.RenderError(f"unsupported {name} form {form}")

    def currency_surface(self, call_id, code, language, name="CURRENCY"):
        """The currency name Python writes next to an amount or as a distractor: the language's DEFAULT currency form."""
        form = next(f for f in self.default_forms("CURRENCY") if self.form_language("CURRENCY", f) == language)
        return self.render_question(call_id, name, code, form)

    def render_money(self, call_id, amount, code, language):
        number = ZH.render_decimal(amount, MONEY_PLACES) if language == "zh" else english_decimal(amount, MONEY_PLACES)
        currency = self.currency_surface(call_id, code, language)
        return number + currency if language == "zh" else f"{number} {currency}"

    def render_rate(self, rate, language):
        return ZH.render_percent(rate, RATE_PLACES) if language == "zh" else english_decimal(rate, RATE_PLACES) + "%"

    # ---- decoding ------------------------------------------------------------------------------------
    def _alias_map(self, name):
        if name not in self._maps:
            profile = self.profiles[PROFILE_OF[name]]
            mapping = {}
            if name == "T01":
                for canonical, languages in (profile.get("language_names") or {}).items():
                    mapping.setdefault(canonical.casefold(), set()).add(canonical)
                    entry = languages.get("zh", {})
                    if entry.get("status") == "APPROVED":
                        mapping.setdefault(entry["surface"].casefold(), set()).add(canonical)
                for canonical in self._product_names():
                    mapping.setdefault(canonical.casefold(), set()).add(canonical)
            else:
                for canonical, aliases in profile["aliases"].items():
                    for alias in [canonical, *aliases]:
                        mapping.setdefault(alias.casefold(), set()).add(canonical)
            self._maps[name] = mapping
        return self._maps[name]

    def _product_names(self):
        return list((self.profiles["synthetic_product_name"].get("language_names") or {}))

    def _lookup(self, name, surface):
        found = self._alias_map(name).get(surface.strip().casefold())
        if not found:
            raise R.RenderError(f"cannot decode {name} surface {surface!r}")
        if len(found) > 1:
            raise R.RenderError(f"{name} surface {surface!r} is ambiguous ({sorted(found)})")
        return next(iter(found))

    def decode_question(self, name, surface):
        if name == "K01":
            return ZH.parse_identifier(surface) if has_han(surface) else R.decode_nric(surface)
        if name == "K02":
            return ZH.parse_date(surface) if has_han(surface) else R.decode_dob(surface)
        if name in ("T03", "T04"):
            return ZH.parse_cardinal(surface) if has_han(surface) else R.decode_number(surface)
        return self._lookup(name, surface)

    def decode_money(self, surface):
        """(amount Decimal, currency code) of a rendered amount-with-currency surface."""
        text = surface.strip()
        if has_han(text):
            names = sorted((alias for alias in self._alias_map("CURRENCY") if has_han(alias)), key=len, reverse=True)
            for alias in names:
                if text.endswith(alias) and len(text) > len(alias):
                    return ZH.parse_decimal(text[:-len(alias)]), self._lookup("CURRENCY", alias)
            raise R.RenderError(f"cannot decode amount {surface!r}")
        number, _, currency = text.partition(" ")
        if not currency:
            raise R.RenderError(f"cannot decode amount {surface!r}")
        return parse_english_decimal(number), self._lookup("CURRENCY", currency)

    def decode_rate(self, surface):
        text = surface.strip()
        if text.endswith("%"):
            return parse_english_decimal(text[:-1])
        return ZH.parse_percent(text)
