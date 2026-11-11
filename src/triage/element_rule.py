"""Element answer from the element model's probabilities: "doubt means column". Written rule, not learned (rule 5).
Spec: spec/features/element_model.md, "Second run" (D52).

A column read as a wall loses the column's +1 raise in the risk matrix, so a column that is merely possible goes to a
person, never to a confident other answer. A wall sent to a person costs time; a column read as a wall can lower a
level (rule 2).
"""

from dataclasses import dataclass

from triage.matrix import CONFIDENCE_THRESHOLD

COLUMN = "column"
UNSURE = "unsure"
COLUMN_DOUBT_DEFAULT = 0.2  # replaced by the value chosen on the validation images (D52); illustrative
COLUMN_WRONG_TARGET = 0.05  # at most this share of columns confidently called something else (illustrative)


@dataclass
class ElementAnswer:
    element: str  # a class name, or "unsure"
    could_be_column: bool  # unsure because a column is possible: a person decides, or Expert needed
    probability: float  # of the answer's class; of the column when could_be_column
    reason: str


def element_answer(probs: dict[str, float], column_doubt: float = COLUMN_DOUBT_DEFAULT,
                   threshold: float = CONFIDENCE_THRESHOLD) -> ElementAnswer:
    """Order: a confident column; any other answer while a column is possible -> unsure; else the best class if it
    reaches the threshold; else unsure."""
    if not probs:
        return ElementAnswer(UNSURE, False, 0.0, "The element model gave no answer: a person decides.")
    best = max(probs, key=probs.get)
    p_col = probs.get(COLUMN, 0.0)
    if best == COLUMN and p_col >= threshold:
        return ElementAnswer(COLUMN, False, p_col, f"The model reads a column ({p_col:.0%}).")
    if best != COLUMN and p_col >= column_doubt:
        return ElementAnswer(UNSURE, True, p_col, f"The model leans to {best} ({probs[best]:.0%}) but a column is "
                                                  f"possible ({p_col:.0%}): a person decides.")
    if probs[best] >= threshold:
        return ElementAnswer(best, False, probs[best], f"The model reads a {best} ({probs[best]:.0%}).")
    return ElementAnswer(UNSURE, False, probs[best], f"The model is unsure ({best}, {probs[best]:.0%}): a person "
                                                     "decides.")


def column_wrong_share(truth: list[str], answers: list[str]) -> float:
    """Share of true columns confidently answered as something else."""
    cols = [a for t, a in zip(truth, answers) if t == COLUMN]
    return sum(a not in (COLUMN, UNSURE) for a in cols) / len(cols) if cols else 0.0


def choose_column_doubt(truth: list[str], probs: list[dict[str, float]], target: float = COLUMN_WRONG_TARGET,
                        threshold: float = CONFIDENCE_THRESHOLD,
                        candidates=tuple(round(0.02 * k, 2) for k in range(1, 26))) -> float:
    """The largest COLUMN_DOUBT (fewest photos sent to a person) that keeps the share of columns confidently called
    something else at or under the target. Run on validation images only, never on the test images."""
    ok = [c for c in candidates
          if column_wrong_share(truth, [element_answer(p, c, threshold).element for p in probs]) <= target]
    return max(ok) if ok else min(candidates)
