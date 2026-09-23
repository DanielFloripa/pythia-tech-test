"""
Identity of a submission, so an identical one arriving while the first is
still running can join it instead of recomputing the same segments.

Lives in domain/ rather than jobs/ because it depends on the request alone,
not on how jobs are stored or executed.
"""

from __future__ import annotations

import hashlib

from pythia_service.domain.models import PredictionRequest


def compute_dedup_key(request: PredictionRequest) -> str:
    """Stable hash of (games, regions, platforms), order-independent.

    Assumes PredictionRequest already stripped and de-duped the lists, so
    sorting is all that is left to do here.
    """
    canonical = repr((
        sorted(request.games),
        sorted(request.regions),
        sorted(request.platforms)
    ))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
