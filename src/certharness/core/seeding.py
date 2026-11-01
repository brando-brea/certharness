import hashlib


def stable_seed(*parts: object) -> int:
    """Process-independent seed (Python's hash() is salted per process)."""
    digest = hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()
    return int(digest[:8], 16)
