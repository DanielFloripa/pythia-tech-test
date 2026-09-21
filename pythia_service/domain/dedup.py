"""
Dedup key: identifica submissões equivalentes de (games, regions, platforms)
para que uma segunda submissão idêntica, enquanto a primeira ainda está
in-flight, reaproveite o mesmo job em vez de reprocessar os mesmos segmentos.

Fica em domain/ (não em jobs/) porque é uma regra de negócio pura — não
depende de como o job é armazenado ou executado, só do request.
"""

from __future__ import annotations

import hashlib

from pythia_service.domain.models import PredictionRequest


def compute_dedup_key(request: PredictionRequest) -> str:
    """Hash determinístico e estável de (games, regions, platforms).

    Determinístico = mesma entrada semântica sempre gera a mesma key,
    independente da ordem em que o cliente enviou as listas.

    Requer que PredictionRequest já tenha normalizado as listas
    (strip + de-dupe, feito no validator do model) — aqui só ordenamos,
    não limpamos dados de novo.
    """
    parts = (
        sorted(request.games),
        sorted(request.regions),
        sorted(request.platforms),
    )
    canonical = repr(parts).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()
