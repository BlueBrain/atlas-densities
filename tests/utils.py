import json


def write_json(filepath, data):
    """Write json file."""
    with open(filepath, "w", encoding="utf-8") as fd:
        json.dump(data, fd, indent=1, separators=(",", ": "))
