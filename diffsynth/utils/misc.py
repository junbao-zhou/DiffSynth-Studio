import os


def get_environment_variable(
    name: str,
    default: str = "",
) -> str:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return value


def parse_non_negative_integer_from_environment(
    name: str,
    default: str,
) -> int:
    raw_value = get_environment_variable(name, default)
    if not raw_value.isdigit():
        raise SystemExit(f"{name} must be a non-negative integer, got: {raw_value}")
    return int(raw_value)


def parse_positive_integer_from_environment(
    name: str,
    default: str,
) -> int:
    raw_value = get_environment_variable(name, default)
    if not raw_value.isdigit():
        raise SystemExit(f"{name} must be a positive integer, got: {raw_value}")
    value = int(raw_value)
    if value < 1:
        raise SystemExit(f"{name} must be >= 1, got: {value}")
    return value
