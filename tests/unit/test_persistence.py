from backend.v2.persistence import JsonRepository


def test_repository_round_trip(tmp_path):
    repo = JsonRepository(str(tmp_path / "state.json"))
    repo.put("students", "s1", {"mastery": 0.7})
    assert repo.get("students", "s1") == {"mastery": 0.7}
