"""Frozen evaluation-profile contract tests."""

import hashlib
from pathlib import Path

import pytest

from vdaworld.eval.profiles import (
    LEWM_ALIGNED_MANIFEST_SHA256,
    LEWM_ALIGNED_N50_V1,
)

REPO_ROOT = Path(__file__).resolve().parents[1]

MANIFEST_PATHS = {
    "pusht": "results/manifests/pusht_lewm_seed42_n50_offset25.json",
    "two_room_swm": "results/manifests/two_room_lewm_seed42_n50_offset25.json",
    "reacher": "results/manifests/reacher_lewm_seed42_n50_offset25.json",
    "cube": "results/manifests/cube_lewm_seed42_n50_offset25.json",
}


def test_lewm_aligned_n50_v1_contract_is_frozen():
    profile = LEWM_ALIGNED_N50_V1
    assert profile.profile_id == "lewm_aligned_n50_v1"
    assert profile.num_eval == 50
    assert profile.goal_offset_steps == 25
    assert profile.horizon == 25
    assert profile.apply_steps == 25
    assert profile.max_episode_steps == 50
    assert profile.cem_population == 300
    assert profile.cem_iters == 30
    assert profile.cem_elite_frac == 0.1
    assert profile.cem_topk == 30
    assert profile.cem_init_std_scale == 1.0
    assert profile.backend == "swm_reference"
    assert profile.solver_implementation == "vdaworld_cem_v1"
    assert "VDA solver implementation" in profile.alignment_scope
    assert set(LEWM_ALIGNED_MANIFEST_SHA256) == {
        "pusht",
        "two_room_swm",
        "reacher",
        "cube",
    }
    assert all(len(digest) == 64 for digest in LEWM_ALIGNED_MANIFEST_SHA256.values())


@pytest.mark.parametrize("benchmark", sorted(MANIFEST_PATHS))
def test_pinned_manifest_hash_matches_manifest_on_disk(benchmark):
    """The pinned digest must track the manifest actually shipped in the repo.

    smoke_p2_mpc.py aborts the lewm_aligned_n50 run on mismatch, so a stale pin
    here surfaces only at launch time for the final campaign. Regenerating a
    manifest without updating the pin must fail here instead.
    """
    manifest_path = REPO_ROOT / MANIFEST_PATHS[benchmark]
    assert manifest_path.is_file(), f"missing manifest: {manifest_path}"
    actual = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    assert actual == LEWM_ALIGNED_MANIFEST_SHA256[benchmark], (
        f"{benchmark}: manifest {manifest_path.name} hashes to {actual} but "
        f"profiles.py pins {LEWM_ALIGNED_MANIFEST_SHA256[benchmark]}. Update the "
        "pin (and the release contract) whenever a manifest is regenerated."
    )
