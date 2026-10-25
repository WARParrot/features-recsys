import pandas as pd

from recsys_corrected_artifact.history import build_histories


def test_histories_support_string_item_ids():
    frame = pd.DataFrame(
        {
            "event_id": [0, 1],
            "user": ["u", "u"],
            "item": ["N1", "N2"],
            "timestamp": [1, 2],
        }
    )
    histories, masks = build_histories(frame, max_hist=2)
    assert histories[0].tolist() == [None, None]
    assert histories[1].tolist() == [None, "N1"]
    assert masks[1].tolist() == [0.0, 1.0]
