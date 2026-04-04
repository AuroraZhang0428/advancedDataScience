"""Configuration values for the apartment leasing agent."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET_PATH = Path("matched_subset_dataset.csv")

MAX_ATTEMPTS = 3
MINIMUM_GOOD_RESULTS = 3
GOOD_SCORE_THRESHOLD = 0.68
TOP_K_RECOMMENDATIONS = 5
SHORTLIST_SIZE = 10


@dataclass(frozen=True)
class ScoringWeights:
    """Weights for deterministic listing scoring."""

    review_rating: float = 0.20
    amenity_match: float = 0.20
    purpose_alignment: float = 0.20
    neighborhood_fit: float = 0.20
    price_score: float = 0.20

    def as_dict(self) -> dict[str, float]:
        """Return weights in plain-dictionary form."""

        return {
            "review_rating": self.review_rating,
            "amenity_match": self.amenity_match,
            "purpose_alignment": self.purpose_alignment,
            "neighborhood_fit": self.neighborhood_fit,
            "price_score": self.price_score,
        }


@dataclass(frozen=True)
class AgentConfig:
    """Bundle application-level defaults in one place."""

    dataset_path: Path = DEFAULT_DATASET_PATH
    max_attempts: int = MAX_ATTEMPTS
    minimum_good_results: int = MINIMUM_GOOD_RESULTS
    good_score_threshold: float = GOOD_SCORE_THRESHOLD
    top_k_recommendations: int = TOP_K_RECOMMENDATIONS
    shortlist_size: int = SHORTLIST_SIZE
    scoring_weights: ScoringWeights = field(default_factory=ScoringWeights)


DEFAULT_CONFIG = AgentConfig()
