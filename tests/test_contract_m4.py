from pathlib import Path

from showvenist.assign.types import BlockParams, DiscSpec
from showvenist.config import load_config
from showvenist.hierarchy.types import DiscNode, DiscRef, FileNode, SeasonNode, ShowNode


def test_tree_helpers():
    f1 = FileNode(Path("a"), 1, 0.0)
    f2 = FileNode(Path("b"), 1, 0.0, ready=False, not_ready_reason="still growing")
    d = DiscNode("d1", DiscRef(1, label="Disc 1"), Path("Disc 1"), [f1, f2], incomplete=True)
    loose = [FileNode(Path("c"), 1, 0.0)]
    s = ShowNode(Path("."), seasons=[SeasonNode(1, Path("S1"), [d], loose=loose)])
    assert d.number == 1 and d.ready_files == [f1]
    assert len(s.all_files()) == 3


def test_block_params_follow_config():
    cfg = load_config(Path("/nonexistent.toml"), env={})
    b = cfg.blocks
    p = BlockParams(b.unclaimed_penalty, b.gap_penalty, b.ordering_threshold)
    assert p == BlockParams()
    assert DiscSpec("d1", (0, 1), 1).fixed is None
