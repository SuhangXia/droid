from __future__ import annotations

from pathlib import Path

import pytest

from fabric_droid.fabric_omni_bridge.encoder import DEFAULT_CHECKPOINT, FabricPhysicalEncoder


def test_final170_read_only_contract() -> None:
    if not DEFAULT_CHECKPOINT.is_file():
        pytest.skip("Final-170 checkpoint is unavailable")
    encoder = FabricPhysicalEncoder(verify_hash=True)
    report = encoder.verify_contract()
    assert report["pass"]
    assert report["preprocessing_contract"]["frames"] == 16
    assert report["token_contract"]["tokens"] == ["M", "S_obs", "R", "C", "U", "global_physical"]
    assert report["writes_to_fabric_omni"] is False
    assert Path(report["checkpoint"]).is_file()
