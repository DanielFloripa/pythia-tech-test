"""Pure rules: status aggregation, the dedup key, and request validation."""

import pytest
from pydantic import ValidationError

from pythia_service.domain.dedup import compute_dedup_key
from pythia_service.domain.models import (
    JobStatus,
    MAX_SEGMENTS_PER_REQUEST,
    PredictionRequest,
    SegmentResult,
    SegmentStatus,
)
from pythia_service.jobs.state import derive_job_status

PENDING, RUNNING = SegmentStatus.PENDING, SegmentStatus.RUNNING
SUCCESS, FAILED = SegmentStatus.SUCCESS, SegmentStatus.FAILED


@pytest.mark.parametrize(
    ("segments", "expected"),
    [
        ([PENDING], JobStatus.PENDING),
        ([PENDING, PENDING], JobStatus.PENDING),
        ([PENDING, RUNNING], JobStatus.RUNNING),
        # Still running: a terminal status must not appear while work is queued.
        ([PENDING, SUCCESS], JobStatus.RUNNING),
        ([PENDING, FAILED], JobStatus.RUNNING),
        ([RUNNING, SUCCESS, FAILED], JobStatus.RUNNING),
        ([SUCCESS, SUCCESS], JobStatus.SUCCESS),
        ([FAILED, FAILED], JobStatus.FAILED),
        ([SUCCESS, FAILED], JobStatus.PARTIAL_SUCCESS),
    ],
)
def test_job_status_is_derived_from_its_segments(segments, expected):
    assert derive_job_status(segments) == expected


def test_empty_segment_list_is_a_bug_not_a_state():
    # A job is always expanded into at least one segment, so this can only be
    # an upstream mistake. It must not degrade into a plausible-looking status.
    with pytest.raises(ValueError, match="empty segment list"):
        derive_job_status([])


def test_dedup_key_ignores_order():
    one = PredictionRequest(games=["gameA", "gameB"], regions=["EU", "NA"], platforms=["iOS"])
    other = PredictionRequest(games=["gameB", "gameA"], regions=["NA", "EU"], platforms=["iOS"])
    assert compute_dedup_key(one) == compute_dedup_key(other)


def test_dedup_key_separates_different_content():
    one = PredictionRequest(games=["gameA"], regions=["EU"], platforms=["iOS"])
    other = PredictionRequest(games=["gameA"], regions=["NA"], platforms=["iOS"])
    assert compute_dedup_key(one) != compute_dedup_key(other)


def test_dedup_key_survives_whitespace_and_repeats():
    clean = PredictionRequest(games=["gameA"], regions=["EU"], platforms=["iOS"])
    messy = PredictionRequest(games=[" gameA ", "gameA"], regions=["EU "], platforms=["iOS"])
    assert compute_dedup_key(clean) == compute_dedup_key(messy)


def test_request_normalizes_input():
    request = PredictionRequest(games=[" gameA ", "gameA"], regions=["EU"], platforms=["iOS", "iOS"])
    assert request.games == ["gameA"]
    assert request.platforms == ["iOS"]


@pytest.mark.parametrize("field", ["games", "regions", "platforms"])
def test_blank_values_are_rejected(field):
    payload = {"games": ["gameA"], "regions": ["EU"], "platforms": ["iOS"], field: ["   "]}
    with pytest.raises(ValidationError, match="empty values"):
        PredictionRequest(**payload)


def test_fan_out_is_capped():
    # 5 x 5 x 3 = 75 segments, over the default cap of 64. Rejected before any
    # job exists, because the cost is the product, not the input size.
    with pytest.raises(ValidationError, match="75 segments"):
        PredictionRequest(
            games=[f"g{i}" for i in range(5)],
            regions=[f"r{i}" for i in range(5)],
            platforms=["p1", "p2", "p3"],
        )


def test_fan_out_at_the_cap_is_accepted():
    request = PredictionRequest(
        games=[f"g{i}" for i in range(MAX_SEGMENTS_PER_REQUEST)],
        regions=["EU"],
        platforms=["iOS"],
    )
    assert len(request.segment_keys()) == MAX_SEGMENTS_PER_REQUEST


def test_segment_keys_expand_the_product_in_a_stable_order():
    request = PredictionRequest(games=["gameA", "gameB"], regions=["EU"], platforms=["iOS", "Android"])
    assert request.segment_keys() == [
        ("gameA", "EU", "iOS"),
        ("gameA", "EU", "Android"),
        ("gameB", "EU", "iOS"),
        ("gameB", "EU", "Android"),
    ]


def test_segment_results_cannot_be_mutated():
    # The store's snapshot is a shallow copy, which is only safe because these
    # objects are replaced rather than edited.
    segment = SegmentResult(game="gameA", region="EU", platform="iOS", status=SegmentStatus.PENDING)
    with pytest.raises(ValidationError):
        segment.status = SegmentStatus.SUCCESS
