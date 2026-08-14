import argparse
import json
import zipfile
from pathlib import Path


def remove_null_quantization_config(value):
    removed = 0
    if isinstance(value, dict):
        if value.get("quantization_config", object()) is None:
            del value["quantization_config"]
            removed += 1
        for child in value.values():
            removed += remove_null_quantization_config(child)
    elif isinstance(value, list):
        for child in value:
            removed += remove_null_quantization_config(child)
    return removed


def repair_archive(source: Path, destination: Path) -> int:
    if source.resolve() == destination.resolve():
        raise ValueError("Destination must differ from source; originals are never overwritten")
    if destination.exists():
        raise FileExistsError(f"Destination already exists: {destination}")

    with zipfile.ZipFile(source, "r") as source_zip:
        config = json.loads(source_zip.read("config.json"))
        removed = remove_null_quantization_config(config)
        if removed == 0:
            raise ValueError("Archive has no null quantization_config fields to repair")

        with zipfile.ZipFile(destination, "x", compression=zipfile.ZIP_DEFLATED) as target_zip:
            for item in source_zip.infolist():
                if item.filename == "config.json":
                    target_zip.writestr(
                        item,
                        json.dumps(config, ensure_ascii=False, separators=(",", ":")),
                    )
                else:
                    target_zip.writestr(item, source_zip.read(item.filename))
    return removed


def main():
    parser = argparse.ArgumentParser(
        description="Create a compatible copy of a Keras archive without changing weights"
    )
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()

    removed = repair_archive(args.source, args.destination)
    print(f"Created {args.destination} after removing {removed} null config fields")


if __name__ == "__main__":
    main()
