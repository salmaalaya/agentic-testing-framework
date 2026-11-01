"""Path helpers must never leak absolute, user-specific paths into reports."""

from agentic_testing.config import PROJECT_ROOT, SCREENSHOTS_DIR, relative_to_root


def test_project_root_contains_the_corpus():
    assert (PROJECT_ROOT / "corpus" / "test_cases.json").is_file()


def test_paths_inside_the_project_become_relative():
    assert relative_to_root(SCREENSHOTS_DIR / "WEB-001.png") == "screenshots/WEB-001.png"


def test_paths_outside_the_project_keep_only_the_file_name(tmp_path):
    assert relative_to_root(tmp_path / "shot.png") == "shot.png"
