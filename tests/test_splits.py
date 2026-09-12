from scripts.create_splits import split_patient_ids


def test_patient_splits_are_reproducible_and_disjoint() -> None:
    ids = [f"patient_{index:03d}" for index in range(20)]
    first = split_patient_ids(ids, 0.7, 0.15, 0.15, seed=42)
    second = split_patient_ids(ids, 0.7, 0.15, 0.15, seed=42)
    assert first == second
    assert len(first["train"]) == 14
    assert len(first["val"]) == 3
    assert len(first["test"]) == 3
    assert not set(first["train"]) & set(first["val"])
    assert set().union(*map(set, first.values())) == set(ids)

