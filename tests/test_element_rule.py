"""The "doubt means column" rule (spec/features/element_model.md, D52)."""

from triage.element_rule import UNSURE, choose_column_doubt, column_wrong_share, element_answer


def test_confident_column():
    a = element_answer({"column": 0.9, "wall": 0.05, "beam": 0.03, "others": 0.02})
    assert a.element == "column" and not a.could_be_column


def test_wall_with_a_possible_column_goes_to_a_person():
    a = element_answer({"wall": 0.7, "column": 0.25, "beam": 0.03, "others": 0.02}, column_doubt=0.2)
    assert a.element == UNSURE and a.could_be_column and "column is possible" in a.reason


def test_even_a_very_confident_wall_yields_to_a_possible_column():
    a = element_answer({"wall": 0.81, "column": 0.19}, column_doubt=0.1)
    assert a.element == UNSURE and a.could_be_column


def test_confident_wall_when_no_column_in_sight():
    a = element_answer({"wall": 0.92, "column": 0.03, "beam": 0.03, "others": 0.02}, column_doubt=0.2)
    assert a.element == "wall" and not a.could_be_column


def test_weak_answer_is_unsure():
    a = element_answer({"wall": 0.5, "beam": 0.4, "column": 0.05, "others": 0.05})
    assert a.element == UNSURE and not a.could_be_column


def test_weak_column_on_top_is_unsure_never_another_element():
    a = element_answer({"column": 0.45, "wall": 0.4, "beam": 0.1, "others": 0.05})
    assert a.element == UNSURE


def test_no_answer():
    assert element_answer({}).element == UNSURE


def test_never_says_safe():
    for p in ({"column": 0.9}, {"wall": 0.9}, {"wall": 0.5, "column": 0.4}, {}):
        assert "safe" not in element_answer(p).reason.lower()


def test_column_wrong_share():
    truth = ["column", "column", "column", "column", "wall"]
    assert column_wrong_share(truth, ["column", "wall", UNSURE, "beam", "column"]) == 0.5


def test_choose_column_doubt_meets_the_target_with_the_least_doubt():
    truth = ["column"] * 4 + ["wall"] * 4
    probs = [{"column": 0.9, "wall": 0.1}, {"column": 0.85, "wall": 0.15},
             {"wall": 0.85, "column": 0.15},  # a column the model calls a wall: needs doubt at 0.15 or less
             {"wall": 0.82, "column": 0.18},  # ... and this one at 0.18 or less
             {"wall": 0.95, "column": 0.05}, {"wall": 0.9, "column": 0.1},
             {"wall": 0.97, "column": 0.03}, {"wall": 0.93, "column": 0.07}]
    c = choose_column_doubt(truth, probs, target=0.0)
    assert c == 0.14  # largest candidate (steps of 0.02) that sends both hidden columns to a person
    answers = [element_answer(p, c).element for p in probs]
    assert column_wrong_share(truth, answers) == 0.0
    assert answers[4:] == ["wall", "wall", "wall", "wall"]  # walls without a real column chance stay answered
