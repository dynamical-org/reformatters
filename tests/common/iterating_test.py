from reformatters.common.iterating import group_by


def test_group_by_parity_preserves_order() -> None:
    items = [3, 1, 4, 2, 5]
    groups = group_by(items, lambda x: x % 2)
    assert groups == ([3, 1, 5], [4, 2])


def test_group_by_empty_returns_empty_tuple() -> None:
    assert group_by([], lambda x: x) == ()
