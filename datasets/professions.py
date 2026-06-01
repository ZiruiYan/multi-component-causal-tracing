from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "professions"


def _read_lines(path):
    return [
        line.strip()
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def get_template_list(path=DATA_DIR):
    return _read_lines(Path(path) / "templates.txt")


def load_word_list(name, path=DATA_DIR):
    return _read_lines(Path(path) / f"{name}_occupations.txt")


word_list_female = load_word_list("female")
word_list_male = load_word_list("male")
