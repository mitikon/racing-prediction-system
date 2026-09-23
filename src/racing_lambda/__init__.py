"""Auditable horse-racing prediction components.

Naming from 2026-09-23:
- 予想4頭抽出方式 = 写真・プロンプト直接判断による新方式（`four_horse_extraction.py`）
- 本格先行予測λ・簡易式先行予測λ・部分空間正則化PCAは実戦成績の悪化を受けて
  2026-09-23に完全削除した。残す判断をした汎用基盤（データ収集・オッズ・
  トリガミ・WSI自己学習・保守専用RSI等）だけが引き続きここにある。
"""

from .backtest import (
    FrozenRaceCase,
    RaceBacktestRow,
    RacingBacktestReport,
    builtin_backtest_cases_2026_09_06,
    run_frozen_backtest,
)
from .evaluation import EvaluationReport, evaluate_prediction
from .four_horse_extraction import (
    EXTRACTION_COUNT,
    EXTRACTION_METHOD,
    ExtractionEvaluation,
    ExtractionResult,
    FourHorseExtraction,
    evaluate_four_horse_extraction,
    freeze_four_horse_extraction,
    settle_four_horse_extraction,
)
from .freeze import freeze_prediction, load_frozen_prediction
from .jra_official_free_ingestion import (
    OfficialSnapshot,
    freeze_snapshot,
    ingest_snapshot,
    ingest_snapshot_file,
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
from .statistical_leading_signal import StatisticalLeadingSignal, StatisticalSignalInput
from .two_layer_leading_signal import TwoLayerSignal, combine_two_layers
from .wsi_self_learning import (
    RACING_WSI_FEATURE_VERSION,
    RACING_WSI_PERIODS,
    RacingWsiOutcomeLearner,
    WsiLearningSummary,
    build_result_labels,
    calculate_support_wsi,
    latest_wsi_state,
)
from .validation_2026_09_06 import (
    RecordedFrozenPrediction,
    RecordedRaceResult,
    ThreeRaceValidationReport,
    VALIDATION_RECORDS_2026_09_06,
    validate_record,
    validation_summary_2026_09_06,
)

__all__ = [
    "AggregateEvidence",
    "ComponentWeights",
    "EvaluationReport",
    "EXTRACTION_COUNT",
    "EXTRACTION_METHOD",
    "ExtractionEvaluation",
    "ExtractionResult",
    "FourHorseExtraction",
    "FrozenRaceCase",
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
    "RACING_WSI_FEATURE_VERSION",
    "RACING_WSI_PERIODS",
    "RaceBacktestRow",
    "RaceDayHorseInput",
    "RacingBacktestReport",
    "RacingWsiOutcomeLearner",
    "RecordedFrozenPrediction",
    "RecordedRaceResult",
    "StatisticalLeadingSignal",
    "StatisticalSignalInput",
    "ThreeRaceValidationReport",
    "TwoLayerSignal",
    "VALIDATION_RECORDS_2026_09_06",
    "WsiLearningSummary",
    "body_weight_fit",
    "build_monthly_condition_stats",
    "build_prediction",
    "build_result_labels",
    "build_snapshot_stats",
    "build_statistical_inputs",
    "builtin_backtest_cases_2026_09_06",
    "calculate_support_wsi",
    "combine_two_layers",
    "evaluate_four_horse_extraction",
    "evaluate_prediction",
    "freeze_four_horse_extraction",
    "freeze_prediction",
    "freeze_snapshot",
    "ingest_snapshot",
    "ingest_snapshot_file",
    "latest_wsi_state",
    "load_frozen_prediction",
    "load_frozen_snapshot",
    "normalized_market_probabilities",
    "pace_position_score",
    "rank_odds_distortion",
    "recent_form_score",
    "run_frozen_backtest",
    "settle_four_horse_extraction",
    "validate_record",
    "validation_summary_2026_09_06",
]
