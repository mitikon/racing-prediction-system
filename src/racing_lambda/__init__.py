"""Auditable horse-racing prediction components.

Naming from 2026-09-09:
- 本格先行予測λ = FullLeadingPredictionLambda
- 簡易式先行予測λ = SimpleLeadingPredictionLambda
- 先行シグナル予測λ remains the separate 部分空間正則化PCA project.
"""

from .backtest import (
    FrozenRaceCase,
    RaceBacktestRow,
    RacingBacktestReport,
    builtin_backtest_cases_2026_09_06,
    run_frozen_backtest,
)
from .evaluation import EvaluationReport, evaluate_prediction
from .freeze import freeze_prediction, load_frozen_prediction
from .full_leading_prediction_lambda import (
    FULL_LEADING_PREDICTION_NAME,
    SIMPLE_LEADING_PREDICTION_NAME,
    FullLeadingPredictionLambda,
    build_jra_training_frame,
    odds_snapshots_from_official,
)
from .jra_official_free_ingestion import (
    OfficialSnapshot,
    freeze_snapshot,
    ingest_snapshot,
    load_frozen_snapshot,
)
from .layer2_live_input import (
    MonthlyConditionStats,
    PastRun,
    RaceDayHorseInput,
    body_weight_fit,
    build_statistical_inputs,
    normalized_market_probabilities,
    pace_position_score,
    recent_form_score,
)
from .monthly_stats_db import (
    AggregateEvidence,
    HorseMonthlyEvidence,
    MonthlyBuildResult,
    MonthlySnapshot,
    build_monthly_condition_stats,
    build_snapshot_stats,
)
from .odds import OddsDistortion, rank_odds_distortion
from .schema import (
    ComponentWeights,
    HorseEntry,
    LeadingSignalPolicy,
    OfficialResult,
    PredictionRow,
    RaceContext,
)
from .scoring import build_prediction
from .simple_leading_signal_v02 import (
    BugType,
    Going,
    SimpleHorseFeatures,
    SimpleLeadingSignalLambdaV02,
    SimplePredictionOutput,
    SimpleRaceContext,
    SimpleScoreBreakdown,
)
from .validation_2026_09_06 import (
    RecordedFrozenPrediction,
    RecordedRaceResult,
    ThreeRaceValidationReport,
    VALIDATION_RECORDS_2026_09_06,
    validate_record,
    validation_summary_2026_09_06,
)

# New explicit public name. Keep the old class exported for backward compatibility
# so existing frozen tests and historical comparisons do not change behavior.
SimpleLeadingPredictionLambda = SimpleLeadingSignalLambdaV02

__all__ = [
    "AggregateEvidence",
    "BugType",
    "ComponentWeights",
    "EvaluationReport",
    "FULL_LEADING_PREDICTION_NAME",
    "FrozenRaceCase",
    "FullLeadingPredictionLambda",
    "Going",
    "HorseEntry",
    "HorseMonthlyEvidence",
    "LeadingSignalPolicy",
    "MonthlyBuildResult",
    "MonthlyConditionStats",
    "MonthlySnapshot",
    "OddsDistortion",
    "OfficialResult",
    "OfficialSnapshot",
    "PastRun",
    "PredictionRow",
    "RaceContext",
    "RaceBacktestRow",
    "RaceDayHorseInput",
    "RacingBacktestReport",
    "RecordedFrozenPrediction",
    "RecordedRaceResult",
    "SIMPLE_LEADING_PREDICTION_NAME",
    "SimpleHorseFeatures",
    "SimpleLeadingPredictionLambda",
    "SimpleLeadingSignalLambdaV02",
    "SimplePredictionOutput",
    "SimpleRaceContext",
    "SimpleScoreBreakdown",
    "ThreeRaceValidationReport",
    "VALIDATION_RECORDS_2026_09_06",
    "body_weight_fit",
    "build_jra_training_frame",
    "builtin_backtest_cases_2026_09_06",
    "build_monthly_condition_stats",
    "build_prediction",
    "build_snapshot_stats",
    "build_statistical_inputs",
    "evaluate_prediction",
    "freeze_prediction",
    "freeze_snapshot",
    "ingest_snapshot",
    "load_frozen_prediction",
    "load_frozen_snapshot",
    "normalized_market_probabilities",
    "odds_snapshots_from_official",
    "pace_position_score",
    "rank_odds_distortion",
    "recent_form_score",
    "run_frozen_backtest",
    "validate_record",
    "validation_summary_2026_09_06",
]
