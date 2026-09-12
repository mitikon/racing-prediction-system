from racing_lambda.torigami_warning import evaluate_torigami, ticket_keys


def test_ticket_counts_for_top5_box_and_axis():
    top5 = ["1", "2", "3", "4", "5"]
    assert len(ticket_keys(top5, bet_type="trifecta")) == 60
    assert len(ticket_keys(top5, bet_type="trio")) == 10
    assert len(ticket_keys(top5, bet_type="trifecta", axis="1")) == 36
    assert len(ticket_keys(top5, bet_type="trio", axis="1")) == 6


def test_trifecta_box_warns_below_60x_break_even():
    top5 = ["1", "2", "3", "4", "5"]
    tickets = ticket_keys(top5, bet_type="trifecta")
    odds = {t: 100.0 for t in tickets}
    odds[tickets[0]] = 59.9
    report = evaluate_torigami(top5, bet_type="trifecta", odds=odds)
    assert report.total_stake == 6000
    assert report.break_even_odds == 60.0
    assert report.torigami_count == 1
    assert report.warning


def test_trio_box_warns_below_10x_break_even():
    top5 = ["1", "2", "3", "4", "5"]
    tickets = ticket_keys(top5, bet_type="trio")
    odds = {tuple(sorted(t)): 20.0 for t in tickets}
    odds[tuple(sorted(tickets[0]))] = 9.9
    report = evaluate_torigami(top5, bet_type="trio", odds=odds)
    assert report.total_stake == 1000
    assert report.break_even_odds == 10.0
    assert report.torigami_count == 1


def test_axis_break_even_and_ticket_counts():
    top5 = ["1", "2", "3", "4", "5"]
    tri_tickets = ticket_keys(top5, bet_type="trifecta", axis="1")
    tri_odds = {t: 40.0 for t in tri_tickets}
    tri_odds[tri_tickets[0]] = 35.9
    tri = evaluate_torigami(top5, bet_type="trifecta", odds=tri_odds, axis="1")
    assert tri.ticket_count == 36
    assert tri.total_stake == 3600
    assert tri.break_even_odds == 36.0
    assert tri.torigami_count == 1

    trio_tickets = ticket_keys(top5, bet_type="trio", axis="1")
    trio_odds = {tuple(sorted(t)): 10.0 for t in trio_tickets}
    trio_odds[tuple(sorted(trio_tickets[0]))] = 5.9
    trio = evaluate_torigami(top5, bet_type="trio", odds=trio_odds, axis="1")
    assert trio.ticket_count == 6
    assert trio.total_stake == 600
    assert trio.break_even_odds == 6.0
    assert trio.torigami_count == 1


def test_missing_odds_are_reported_not_assumed_safe():
    report = evaluate_torigami(
        ["1", "2", "3", "4", "5"], bet_type="trio", odds={}
    )
    assert report.odds_available == 0
    assert report.missing_odds_count == 10
    assert not report.warning
