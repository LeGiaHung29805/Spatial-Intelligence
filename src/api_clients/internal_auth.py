import hmac


def has_valid_internal_token(configured_token: str, provided_token: str | None) -> bool:
    return bool(configured_token and provided_token) and hmac.compare_digest(
        configured_token, provided_token
    )
