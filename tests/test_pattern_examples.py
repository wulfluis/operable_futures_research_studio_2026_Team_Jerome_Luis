from triage.matrix import PATTERNS
from viz.pattern_examples import COMMONS, EXAMPLES


def test_every_pattern_of_the_matrix_has_photos_in_matrix_order():
    assert [p for p, *_ in EXAMPLES] == PATTERNS
    assert all(1 <= len(photos) <= 3 for _, photos, _ in EXAMPLES)


def test_every_commons_photo_has_a_file_name_an_author_and_a_licence():
    used = {ref for _, photos, _ in EXAMPLES for src, ref in photos if src == "commons"}
    assert used <= set(COMMONS)
    for name, author, licence in COMMONS.values():
        assert name and author and licence


def test_every_column_says_how_sure_it_is():
    assert all(basis.strip() for _, _, basis in EXAMPLES)
