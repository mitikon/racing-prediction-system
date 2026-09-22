from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from racing_lambda.race_day_collection import RaceDayCollectionPolicy, ScheduledRace


JST = ZoneInfo("Asia/Tokyo")
DAY = datetime(2026, 9, 27, 10, 10, tzinfo=JST)


def _race(number=1, kind="flat", status="official", start=DAY):
    return ScheduledRace(
        race_id=f"20260927-NAKAYAMA-{number:02d}", venue="中山",
        race_number=number, scheduled_start=start, race_type=kind,
        result_status=status,
    )


def test_collects_official_flat_result_once_after_grace_period():
    policy = RaceDayCollectionPolicy()
    now = DAY + timedelta(minutes=6)
    assert policy.eligible(_race(), now=now)
    assert not policy.eligible(_race(), now=now, already_collected={_race().race_id})


def test_excludes_jump_race_and_unconfirmed_result():
    policy = RaceDayCollectionPolicy()
    now = DAY + timedelta(minutes=20)
    assert not policy.eligible(_race(kind="jump"), now=now)
    assert not policy.eligible(_race(status="provisional"), now=now)


def test_operates_only_on_same_meeting_day_and_daytime_window():
    policy = RaceDayCollectionPolicy()
    assert not policy.eligible(_race(), now=DAY + timedelta(days=1))
    assert not policy.eligible(_race(), now=DAY.replace(hour=8, minute=59))
    late_race = _race(number=12, start=DAY.replace(hour=16, minute=20))
    assert policy.eligible(late_race, now=DAY.replace(hour=16, minute=30))
