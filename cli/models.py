from enum import StrEnum


class AnalystType(StrEnum):
    MARKET = "market"
    # Wire value stays "social" for saved-config and string-keyed-caller
    # back-compat; the user-facing label is "Sentiment Analyst".
    SOCIAL = "social"
    NEWS = "news"
    FUNDAMENTALS = "fundamentals"
    OPTIONS = "options"
    CALENDAR = "calendar"


class AssetType(StrEnum):
    STOCK = "stock"
    CRYPTO = "crypto"
