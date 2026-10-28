from __future__ import annotations

import re

STUDENT_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


def validate_student_id(student_id: str) -> None:
    if not STUDENT_ID_PATTERN.fullmatch(student_id) or student_id == "Unknown":
        raise ValueError(
            "Student IDs must be 1-64 safe ASCII letters, digits, '.', '_' or '-'; "
            "'Unknown' is reserved."
        )
