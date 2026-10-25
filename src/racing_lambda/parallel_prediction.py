"""1回のレースデータ投入から、複数の構想を並行予想として同時生成する。

これまでTop5(scoring.py)・軸形成(axis_formation.py)・展開予想
(pace_scenario.py)・レース選別(race_selection.py)はそれぞれ独立した
モジュールで、同じレースのデータを別々に入力する必要があった。この
モジュールは1回のデータ投入(RaceIntake)から全ての並行予想を生成し、
同じ入力・同じ発走前時刻で固定保存することで、後からどの構想が実際
に優れていたかを完全に同じレース集合で比較できるようにする。

予測ロジック自体(scoring.build_prediction・各構想のアルゴリズム)は
一切変更しない。このモジュールは入力を配り、出力を束ねるだけの境界
である。
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from hashlib import sha256
from pathlib import Path
from typing import Any, Mapping

from .axis_formation import AxisFormationPrediction
from .pace_scenario import PaceScenarioAnalysis, RunningStyle, analyze_pace_scenario
from .race_selection import RaceEdgeCandidacy, score_race_edge_candidacy
from .schema import ComponentWeights, HorseEntry, PredictionRow
from .scoring import build_prediction


@dataclass(frozen=True)
class RaceIntake:
    """並行予想の唯一の入力。全構想がこの1つのデータから導出される。"""

    race_id: str
    race_name: str
    scheduled_start: str
    frozen_at: str
    horses: tuple[HorseEntry, ...]
    running_styles: Mapping[str, RunningStyle] | None = None
    odds_by_horse: Mapping[str, float] | None = None

    def __post_init__(self) -> None:
        if not self.race_id.strip():
            raise ValueError("race_id is required")
        if len(self.horses) < 4:
            raise ValueError("at least four horses are required for axis-formation extraction")
        horse_ids = {entry.horse_id for entry in self.horses}
        if len(horse_ids) != len(self.horses):
            raise ValueError("horse_id values must be unique")
        if self.running_styles is not None and set(self.running_styles) != horse_ids:
            raise ValueError(
                "running_styles must cover exactly the same horses as `horses`, or be omitted entirely - "
                "a partial set would misrepresent the field's pace composition"
            )


@dataclass(frozen=True)
class ParallelPredictionBundle:
    race_id: str
    race_name: str
    scheduled_start: str
    frozen_at: str
    base_prediction: tuple[PredictionRow, ...]
    pace_scenario_analysis: PaceScenarioAnalysis | None
    pace_adjusted_prediction: tuple[PredictionRow, ...] | None
    axis_formation: AxisFormationPrediction
    edge_candidacy: RaceEdgeCandidacy | None
    schema_version: int = 1

    @property
    def primary_prediction(self) -> tuple[PredictionRow, ...]:
        """展開調整済みの予想があればそれを、無ければ基本予想を使う。

        軸形成・レース選別はこちらを入力として使う - 展開理論を使える
        状況では、それを反映した方を「今回の代表予想」として扱う。
        """
        return self.pace_adjusted_prediction or self.base_prediction


def build_parallel_prediction(
    intake: RaceIntake,
    weights: ComponentWeights | None = None,
) -> ParallelPredictionBundle:
    """1つのRaceIntakeから、全ての並行予想構想を一度に生成する。"""
    base_prediction = tuple(build_prediction(intake.horses, weights))

    pace_analysis: PaceScenarioAnalysis | None = None
    pace_adjusted_prediction: tuple[PredictionRow, ...] | None = None
    if intake.running_styles is not None:
        pace_analysis = analyze_pace_scenario(intake.running_styles)
        adjusted_horses = tuple(
            replace(entry, pace_fit=pace_analysis.pace_fit_by_horse[entry.horse_id])
            for entry in intake.horses
        )
        pace_adjusted_prediction = tuple(build_prediction(adjusted_horses, weights))

    primary = pace_adjusted_prediction or base_prediction
    primary_top5 = tuple(row.horse_id for row in sorted(primary, key=lambda row: row.rank)[:5])
    if len(primary_top5) < 4:
        raise ValueError("at least four ranked horses are required for axis-formation extraction")

    axis_formation = AxisFormationPrediction(
        race_id=intake.race_id,
        scheduled_start=intake.scheduled_start,
        frozen_at=intake.frozen_at,
        axis_horse_id=primary_top5[0],
        contender_horse_ids=primary_top5[1:4],
        rationale=(
            "pace-adjusted lambda rank 1-4"
            if pace_adjusted_prediction is not None
            else "base lambda rank 1-4"
        ),
    )

    edge_candidacy: RaceEdgeCandidacy | None = None
    if intake.odds_by_horse:
        edge_candidacy = score_race_edge_candidacy(
            race_id=intake.race_id,
            race_name=intake.race_name,
            lambda_top5=primary_top5,
            odds_by_horse=intake.odds_by_horse,
        )

    return ParallelPredictionBundle(
        race_id=intake.race_id,
        race_name=intake.race_name,
        scheduled_start=intake.scheduled_start,
        frozen_at=intake.frozen_at,
        base_prediction=base_prediction,
        pace_scenario_analysis=pace_analysis,
        pace_adjusted_prediction=pace_adjusted_prediction,
        axis_formation=axis_formation,
        edge_candidacy=edge_candidacy,
    )


def _canonical(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")


def freeze_parallel_prediction(root: str | Path, bundle: ParallelPredictionBundle) -> Path:
    """並行予想一式を発走前に書き込み禁止で固定保存する。"""
    path = Path(root) / bundle.race_id / "PARALLEL_PREDICTION" / "prediction.json"
    if path.exists():
        raise FileExistsError(f"immutable parallel-prediction record already exists: {path}")
    payload: dict[str, Any] = {
        "race_id": bundle.race_id,
        "race_name": bundle.race_name,
        "scheduled_start": bundle.scheduled_start,
        "frozen_at": bundle.frozen_at,
        "schema_version": bundle.schema_version,
        "base_prediction": [asdict(row) for row in bundle.base_prediction],
        "pace_scenario_analysis": asdict(bundle.pace_scenario_analysis) if bundle.pace_scenario_analysis else None,
        "pace_adjusted_prediction": (
            [asdict(row) for row in bundle.pace_adjusted_prediction]
            if bundle.pace_adjusted_prediction is not None
            else None
        ),
        "axis_formation": asdict(bundle.axis_formation),
        "edge_candidacy": asdict(bundle.edge_candidacy) if bundle.edge_candidacy else None,
    }
    payload["checksum_sha256"] = sha256(_canonical(payload)).hexdigest()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write("\n")
    return path


def load_parallel_prediction(path: str | Path) -> Mapping[str, Any]:
    """固定保存した並行予想一式を読み込み、チェックサムを検証する。

    各構想のdataclassへの完全な再構築ではなく、検証済みの辞書を返す -
    呼び出し側が必要な構想だけを取り出せば十分であり、不要な再構築は
    しない。
    """
    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise ValueError("parallel-prediction record must be a regular non-symlink file")
    payload = json.loads(source.read_text(encoding="utf-8"))
    checksum = payload.pop("checksum_sha256")
    actual = sha256(_canonical(payload)).hexdigest()
    if checksum != actual:
        raise ValueError("parallel-prediction checksum mismatch: file may have been tampered with")
    return payload
