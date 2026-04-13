#!/usr/bin/env python3
"""Download upstream Marathi NER datasets.

The initial sources are the L3Cube Marathi NER datasets hosted inside one
GitHub repository. The source registry is intentionally explicit so additional
sources can be added without changing the CLI contract.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import subprocess
import sys
import urllib.request
from pathlib import Path
from typing import Literal


SourceKind = Literal["git_sparse", "direct_url"]


@dataclasses.dataclass(frozen=True)
class DatasetSource:
    name: str
    kind: SourceKind
    url: str
    output_dir: Path
    license: str
    redistribution: str
    citation: str
    sparse_paths: tuple[str, ...] = ()
    files: tuple[str, ...] = ()


SOURCES: tuple[DatasetSource, ...] = (
    DatasetSource(
        name="l3cube-mahaner",
        kind="git_sparse",
        url="https://github.com/l3cube-pune/MarathiNLP.git",
        output_dir=Path("l3cube-marathinlp"),
        sparse_paths=("L3Cube-MahaNER/IOB", "L3Cube-MahaNER/README.md", "README.md"),
        files=(
            "L3Cube-MahaNER/IOB/train_iob.txt",
            "L3Cube-MahaNER/IOB/valid_iob.txt",
            "L3Cube-MahaNER/IOB/test_iob.txt",
        ),
        license="CC BY-NC-SA 4.0",
        redistribution="Research use; non-commercial ShareAlike license per upstream README.",
        citation="Litake et al., L3Cube-MahaNER, WILDRE 2022",
    ),
    DatasetSource(
        name="l3cube-mahasocialner",
        kind="git_sparse",
        url="https://github.com/l3cube-pune/MarathiNLP.git",
        output_dir=Path("l3cube-marathinlp"),
        sparse_paths=(
            "L3Cube-MahaSocialNER/IOB",
            "L3Cube-MahaSocialNER/README.md",
            "README.md",
        ),
        files=(
            "L3Cube-MahaSocialNER/IOB/train_iob.txt",
            "L3Cube-MahaSocialNER/IOB/valid_iob.txt",
            "L3Cube-MahaSocialNER/IOB/test_iob.txt",
        ),
        license="Unresolved in upstream top-level license sentence",
        redistribution=(
            "Treat as unresolved for publication until upstream confirms terms; "
            "source is publicly downloadable from the MarathiNLP repository."
        ),
        citation="Chaudhari et al., L3Cube-MahaSocialNER, arXiv 2023",
    ),
)


def run(command: list[str], cwd: Path | None = None) -> None:
    print("+", " ".join(command), file=sys.stderr)
    subprocess.run(command, cwd=cwd, check=True)


def selected_sources(names: list[str]) -> list[DatasetSource]:
    by_name = {source.name: source for source in SOURCES}
    if not names:
        return list(SOURCES)

    unknown = sorted(set(names) - set(by_name))
    if unknown:
        available = ", ".join(sorted(by_name))
        raise SystemExit(f"Unknown source(s): {', '.join(unknown)}. Available: {available}")

    return [by_name[name] for name in names]


def clone_or_update_sparse_repo(repo_url: str, destination: Path, sparse_paths: set[str]) -> None:
    if destination.exists() and not (destination / ".git").exists():
        raise SystemExit(f"{destination} exists but is not a Git repository")

    if not destination.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        run(
            [
                "git",
                "clone",
                "--depth",
                "1",
                "--filter=blob:none",
                "--sparse",
                repo_url,
                str(destination),
            ]
        )
    else:
        run(["git", "checkout", "main"], cwd=destination)
        run(["git", "pull", "--ff-only", "--depth", "1", "origin", "main"], cwd=destination)

    run(["git", "sparse-checkout", "set", *sorted(sparse_paths)], cwd=destination)


def download_direct_files(source: DatasetSource, root: Path) -> None:
    if not source.files:
        raise SystemExit(f"{source.name} does not define any files to download")

    target_dir = root / source.output_dir
    target_dir.mkdir(parents=True, exist_ok=True)
    for relative_file in source.files:
        output_path = target_dir / relative_file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        source_url = source.url.rstrip("/") + "/" + relative_file.lstrip("/")
        print(f"Downloading {source_url} -> {output_path}", file=sys.stderr)
        urllib.request.urlretrieve(source_url, output_path)


def validate_files(root: Path, sources: list[DatasetSource]) -> dict[str, dict[str, object]]:
    manifest: dict[str, dict[str, object]] = {}
    for source in sources:
        base_dir = root / source.output_dir
        missing = [path for path in source.files if not (base_dir / path).is_file()]
        if missing:
            raise SystemExit(f"{source.name} is missing expected file(s): {', '.join(missing)}")

        manifest[source.name] = {
            "kind": source.kind,
            "url": source.url,
            "output_dir": str(source.output_dir),
            "files": source.files,
            "license": source.license,
            "redistribution": source.redistribution,
            "citation": source.citation,
        }
    return manifest


def write_manifest(path: Path, manifest: dict[str, dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        action="append",
        default=[],
        help="Source name to download. May be repeated. Defaults to all sources.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/raw/upstream"),
        help="Directory where upstream data checkouts/downloads are stored.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/raw/source_manifest.json"),
        help="Path for the generated source manifest JSON.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    sources = selected_sources(args.source)

    sparse_by_repo: dict[tuple[str, Path], set[str]] = {}
    direct_sources: list[DatasetSource] = []
    for source in sources:
        if source.kind == "git_sparse":
            key = (source.url, args.output_dir / source.output_dir)
            sparse_by_repo.setdefault(key, set()).update(source.sparse_paths)
        elif source.kind == "direct_url":
            direct_sources.append(source)
        else:
            raise AssertionError(f"Unsupported source kind: {source.kind}")

    for (repo_url, destination), sparse_paths in sparse_by_repo.items():
        clone_or_update_sparse_repo(repo_url, destination, sparse_paths)

    for source in direct_sources:
        download_direct_files(source, args.output_dir)

    manifest = validate_files(args.output_dir, sources)
    write_manifest(args.manifest, manifest)
    print(f"Wrote {args.manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
