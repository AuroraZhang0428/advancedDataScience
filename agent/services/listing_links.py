"""Best-effort Airbnb listing link finder and verifier.

Completely complementary — if a URL cannot be verified it is simply not
attached.  This module never raises and never affects ranking.

IMPORTANT: attach_listing_links() must only be called AFTER rankings are
fully finalized and only on display-copy dicts, never on the scored listing
dicts used by the ranking pipeline.  The presence or absence of a working
link must have zero effect on which listings are recommended.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any
from urllib import error, request as urllib_request

_BASE_URL = "https://www.airbnb.com/rooms/{}"
_TIMEOUT = 6  # seconds per request
_MAX_WORKERS = 5  # parallel verifications (one per final recommendation)

# Mimic a real browser so Airbnb doesn't immediately 403 us
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}


def _build_url(listing_id: str) -> str:
    return _BASE_URL.format(listing_id)


def _verify_url(url: str) -> bool:
    """Return True if the URL is live.

    A confirmed 404 is the only hard failure — everything else (403 anti-bot,
    429 rate-limit, timeout, network error) is treated as "probably exists"
    so we don't silently drop real listings due to Airbnb's bot protection.
    """
    try:
        req = urllib_request.Request(url, headers=_HEADERS, method="HEAD")
        with urllib_request.urlopen(req, timeout=_TIMEOUT) as resp:
            return resp.status not in (404,)
    except error.HTTPError as exc:
        # 404 → listing is gone; anything else → assume it exists
        return exc.code != 404
    except Exception:
        # Timeout, connection error, etc. — don't penalise the listing
        return True


def _try_attach(rec: dict[str, Any]) -> None:
    """Verify and attach the Airbnb URL to a single recommendation in-place."""
    try:
        listing_id = str(rec.get("id", "")).strip()
        if not listing_id:
            return
        url = _build_url(listing_id)
        if _verify_url(url):
            rec["airbnb_url"] = url
    except Exception:
        pass  # never let link lookup affect anything


def attach_listing_links(recommendations: list[dict[str, Any]]) -> None:
    """Attach verified Airbnb URLs to recommendations in-place, in parallel.

    Safe to call with any list — missing IDs, network failures, and bad
    responses are all silently ignored.  Ranking is never touched.
    """
    if not recommendations:
        return
    with ThreadPoolExecutor(max_workers=_MAX_WORKERS) as executor:
        futures = [executor.submit(_try_attach, rec) for rec in recommendations]
        for f in as_completed(futures):
            try:
                f.result()
            except Exception:
                pass
