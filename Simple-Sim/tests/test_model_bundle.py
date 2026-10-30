"""Test bundle.json maintenance."""

import json
import warnings

from simple_sim.model_bundle import BUNDLE_META_NAME, read_bundle_meta, upsert_bundle_meta


def test_corrupt_bundle_meta_is_backed_up_and_rebuilt(tmp_path):
    bundle = tmp_path / "m.bundle"
    bundle.mkdir()
    (bundle / "prof_a@1.pt").write_bytes(b"not a real checkpoint")
    (bundle / "prof_a@1_last.pt").write_bytes(b"x")
    (bundle / "prof_b@1.pt").write_bytes(b"x")
    (bundle / BUNDLE_META_NAME).write_text("{ broken json", encoding="utf-8")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        upsert_bundle_meta(bundle, "prof_b@1", bundle / "prof_b@1.pt")
    assert any("Corrupt" in str(w.message) for w in caught)

    backups = list(bundle.glob(f"{BUNDLE_META_NAME}.corrupt-*"))
    assert len(backups) == 1
    meta = read_bundle_meta(bundle)
    assert meta is not None
    assert meta.checkpoints == {"prof_a@1": "prof_a@1.pt", "prof_b@1": "prof_b@1.pt"}
    json.loads((bundle / BUNDLE_META_NAME).read_text(encoding="utf-8"))
