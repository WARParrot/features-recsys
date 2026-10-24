import json

import pandas as pd

from compare_methods import main


def test_kion_rolling_fixture_runs_through_cli(monkeypatch, tmp_path, capsys):
    rows = []
    for day in range(24):
        for user in range(1, 101):
            rows.append(
                {
                    "user_id": user,
                    "item_id": 10 + ((user + day) % 4),
                    "last_watch_dt": f"2021-01-{day + 1:02d}",
                    "total_dur": 100,
                    "watched_pct": 80 if (user + day) % 2 else 20,
                }
            )
    interactions = tmp_path / "interactions.csv"
    pd.DataFrame(rows).to_csv(interactions, index=False)

    monkeypatch.setattr(
        "sys.argv",
        [
            "compare_methods.py", "--kion-interactions", str(interactions),
            "--evaluation", "rolling", "--folds", "1", "--seeds", "42",
            "--forward-max-candidates", "1", "--forward-steps", "1", "--json",
        ],
    )
    main()
    payload = json.loads(capsys.readouterr().out)
    assert payload["metadata"]["dataset"] == "kion"
    assert payload["metadata"]["data_summary"]["folds"] == 1
    assert payload["runs"][0]["folds"][0]["split_metadata"]["test"]["rows"] > 0
