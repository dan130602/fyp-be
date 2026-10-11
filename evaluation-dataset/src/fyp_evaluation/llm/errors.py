"""Transport error taxonomy.

The workflow layer never inspects provider exceptions: transports translate them into these classes.
"""


# Provider codes meaning "no credits/quota": a billing condition that no retry can fix.
QUOTA_CODES = {"insufficient_quota", "credit_balance_exhausted", "billing_hard_limit_reached"}


class TransportError(Exception):
    """Base class for every error a transport may raise."""

    fatal = False


class CredentialsError(TransportError):
    """No usable API key; raised before any request is made."""

    fatal = True


class FatalRequestError(TransportError):
    """Permanent configuration/schema/authentication failure; the whole run must stop (never retried)."""

    fatal = True


class RetryableError(TransportError):
    """Rate limit, timeout or 5xx that was definitely not processed or whose retry is harmless."""


class AmbiguousError(TransportError):
    """The outcome of a non-idempotent request (batch creation) is unknown; the request may have been accepted."""
