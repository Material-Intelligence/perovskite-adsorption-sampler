#!/usr/bin/env python
"""Repository hygiene scanner.

The scanner walks the working tree and reports what does not belong in a portable source
tree:

* absolute filesystem paths, which a repository that addresses its own files relatively
  never needs;
* scheduler account names in submission templates that are not angle-bracketed placeholders;
* e-mail addresses other than the ones in :data:`ALLOWED_EMAILS`;
* credential-shaped tokens;
* non-ASCII text, since this project is written in English;
* files larger than :data:`MAX_FILE_BYTES`;
* machine-learned-potential checkpoints, which are downloaded from their publisher rather
  than redistributed here.

A file of extra literal terms may be supplied with ``--deny-file`` or through the
``PEROVML_RELEASE_DENY`` environment variable; it is read from outside the scanned tree, so
no configuration file is needed inside the repository.

The check is deliberately fail-closed: it prints every match and exits non-zero if there is
at least one, so a legitimate match has to be rewritten rather than silenced. Only the Python
standard library is used, so the scanner runs in a bare environment and can be wired into CI
before the package itself is installed.

The scanner exempts its own source file from the path and account rules, whose regular
expressions are spelled out here; every other rule (size, suffix, non-ASCII, e-mail) still
applies to it.

Usage:
    python tools/release_check.py                     # scan the repository root, fail on any hit
    python tools/release_check.py --skip-cjk          # do not report non-ASCII CJK text
    python tools/release_check.py --deny-file FILE    # also search for the terms listed in FILE
    python tools/release_check.py --json              # machine-readable findings on stdout
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

MAX_FILE_BYTES = 5 * 1024 * 1024
"""Largest file size, in bytes, that a released file is allowed to have."""

MAX_TEXT_LENGTH = 120
"""Matched text longer than this is truncated in the report."""

SKIP_DIRS = frozenset(
    {
        ".git",
        ".venv",
        "venv",
        "env",
        "__pycache__",
        ".pytest_cache",
        ".ruff_cache",
        ".mypy_cache",
        ".idea",
        ".vscode",
        "build",
        "dist",
        "node_modules",
    }
)
"""Directory names that are never part of a release and are not walked into."""

CHECKPOINT_SUFFIXES = frozenset({".pt", ".pth", ".npz", ".h5", ".ckpt"})
"""Suffixes of model weights and array dumps that must not be committed."""

SELF_RELPATH = "tools/release_check.py"
"""Path of this file relative to the repository root; exempt from the disclosure patterns only."""

PLACEHOLDER_ROOT = "path/to/"
"""The one absolute-path prefix that is accepted, because it is an obvious documentation stub."""

DENY_FILE_ENV_VAR = "PEROVML_RELEASE_DENY"
"""Environment variable naming a file of extra literal terms, used when ``--deny-file`` is not given.

There is deliberately no in-tree default: the term list belongs to whoever runs the scan, not to
the repository, so it is never looked for inside the scanned tree.
"""

ALLOWED_EMAILS = frozenset({"xiejh.mail@gmail.com"})
"""E-mail addresses allowed to appear in the tree. Every other address is reported."""

EMAIL_PATTERN = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
"""Matches an e-mail address, so that collaborator addresses and template stubs are caught."""

CJK_ALLOWLIST: frozenset = frozenset()
"""Relative paths allowed to contain CJK characters. Kept empty: this project is written in English."""

CJK_RANGES: Tuple[Tuple[int, int], ...] = ((0x4E00, 0x9FFF), (0x3000, 0x303F), (0xFF00, 0xFFEF))
"""Code-point ranges treated as CJK: ideographs, CJK punctuation, and fullwidth/halfwidth forms.

The ranges are given as integers rather than as literal characters so that this file, which is
itself scanned, stays pure ASCII.
"""

CJK_PATTERN = re.compile("[{}]+".format("".join(f"{chr(low)}-{chr(high)}" for low, high in CJK_RANGES)))
"""Matches one or more consecutive CJK characters."""

CLUSTER_ROOTS = (
    "pscratch",
    "scratch",
    "global",
    "gpfs",
    "lustre",
    "nfs",
    "mnt",
    "work",
    "projects",
    "project",
)
"""Directory names that start an absolute path on a shared HPC filesystem rather than a laptop.

A released tree addresses its own files relatively, so any absolute path under one of these
roots is a leftover from the machine the code was developed on.
"""

DISCLOSURE_PATTERNS: Tuple[Tuple[str, "re.Pattern[str]"], ...] = (
    ("cluster-path", re.compile(r"(?<![\w./-])/(?:{})(?:/[\w.+-]+)+".format("|".join(CLUSTER_ROOTS)))),
    ("home-path", re.compile(r"(?<![\w./-])/(?:Users|home)/(?!<)[\w.+-]+")),
    ("potcar-path", re.compile(rf"(?<![\w./-])/(?!{PLACEHOLDER_ROOT})(?:[\w.+-]+/)+potpaw\w*", re.IGNORECASE)),
    ("slurm-account", re.compile(r"--account[=\s]+(?!<)[^\s\"']+")),
    ("slurm-account", re.compile(r"^#SBATCH\b.*?(?<![\w-])-A\s+(?!<)\S+")),
)
"""Rules for machine-specific paths and accounts, as ``(category, pattern)`` pairs.

Every rule matches a shape rather than one site's names: an absolute path under a shared
filesystem root or a home directory, a PAW potential directory, and a scheduler account that is
not an angle-bracketed ``<PLACEHOLDER>``. Anything narrower than a shape is left to the optional
term file named by ``--deny-file``.
"""

SECRET_PREFIX_PATTERN = re.compile(
    r"\b(?:sk-[A-Za-z0-9_-]{16,}"
    r"|gh[pousr]_[A-Za-z0-9]{16,}"
    r"|AKIA[0-9A-Z]{12,}"
    r"|hf_[A-Za-z0-9]{16,}"
    r"|xox[abprs]-[A-Za-z0-9-]{16,})"
)
"""Tokens whose vendor prefix identifies them as credentials outright."""

LONG_TOKEN_PATTERN = re.compile(r"(?<![A-Za-z0-9])[A-Za-z0-9]{32,}(?![A-Za-z0-9])")
"""Candidate credentials: unbroken alphanumeric runs of 32 characters or more."""


@dataclass(frozen=True)
class Finding:
    """A single release blocker found in the tree.

    Attributes:
        path: Path of the offending file, relative to the scanned root.
        line: 1-based line number, or 0 when the finding concerns the file itself (its size,
            its suffix or its name) rather than its contents.
        category: Short slug naming the rule that fired, e.g. ``"allocation"``.
        text: The matched text, truncated to :data:`MAX_TEXT_LENGTH` characters.
    """

    path: str
    line: int
    category: str
    text: str

    def __str__(self) -> str:
        """Return the finding in ``path:line:matched-text`` form."""
        return f"{self.path}:{self.line}:{self.text}"

    def as_dict(self) -> Dict[str, object]:
        """Return an MSONable-style dict representation.

        Returns:
            A JSON-serialisable dict carrying ``@module``/``@class`` keys alongside the fields.
        """
        return {
            "@module": type(self).__module__,
            "@class": type(self).__name__,
            "path": self.path,
            "line": self.line,
            "category": self.category,
            "text": self.text,
        }

    @classmethod
    def from_dict(cls, dct: Dict[str, object]) -> "Finding":
        """Reconstruct a finding from :meth:`as_dict` output.

        Args:
            dct: A dict produced by :meth:`as_dict`.

        Returns:
            The reconstructed :class:`Finding`.
        """
        return cls(
            path=str(dct["path"]),
            line=int(dct["line"]),  # type: ignore[arg-type]
            category=str(dct["category"]),
            text=str(dct["text"]),
        )


def looks_like_secret(token: str) -> bool:
    """Judge whether an alphanumeric token is credential-shaped.

    Two shapes are treated as credentials: a long hexadecimal digest, and a long mixed-case
    token that also carries digits. Ordinary prose, identifiers and file names almost never
    satisfy either, which keeps the false-positive rate low without weakening the check.

    Args:
        token: An unbroken alphanumeric run of at least 32 characters.

    Returns:
        True if the token should be reported as a possible credential.
    """
    if re.fullmatch(r"[0-9a-fA-F]{32,}", token) and any(char.isdigit() for char in token):
        return True
    has_digit = any(char.isdigit() for char in token)
    has_lower = any(char.islower() for char in token)
    has_upper = any(char.isupper() for char in token)
    return has_digit and has_lower and has_upper


def load_deny_terms(path: Optional[Path]) -> Tuple[str, ...]:
    """Read the extra literal terms to search for.

    The file holds one term per line; blank lines and lines whose first non-blank character is
    ``#`` are ignored. Terms are matched case-insensitively as plain substrings, not as regular
    expressions, so they need no escaping.

    Args:
        path: The deny-list file, or None when no deny-list is in use.

    Returns:
        The terms, lower-cased, in file order.

    Raises:
        FileNotFoundError: If ``path`` is given but does not exist.
    """
    if path is None:
        return ()
    terms: List[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            terms.append(stripped.lower())
    return tuple(terms)


def resolve_deny_file(root: Path, explicit: Optional[Path]) -> Optional[Path]:
    """Decide which file of extra literal terms to use, if any.

    The order is ``--deny-file``, then :data:`DENY_FILE_ENV_VAR`. Nothing inside ``root`` is
    consulted, so the scanned tree never supplies its own term list.

    Args:
        root: Root of the scanned tree. Unused; kept so that the signature stays stable for
            callers that scan a tree other than the current directory.
        explicit: The path passed on the command line, or None.

    Returns:
        The file of extra terms to read, or None when the rule is not in use.
    """
    del root  # The term list is never taken from the scanned tree.
    if explicit is not None:
        return explicit
    from_env = os.environ.get(DENY_FILE_ENV_VAR)
    return Path(from_env) if from_env else None


def _truncate(text: str) -> str:
    """Shorten matched text so one finding stays on one line.

    Args:
        text: The raw matched text.

    Returns:
        The text, collapsed to a single line and truncated to :data:`MAX_TEXT_LENGTH`.
    """
    flat = " ".join(text.split())
    if len(flat) > MAX_TEXT_LENGTH:
        return flat[:MAX_TEXT_LENGTH] + "..."
    return flat


def iter_files(root: Path) -> Iterator[Path]:
    """Walk the tree, skipping the directories in :data:`SKIP_DIRS`.

    Args:
        root: Directory to walk.

    Yields:
        Every regular file below ``root`` that is part of the release tree.
    """
    for path in sorted(root.rglob("*")):
        if any(part in SKIP_DIRS for part in path.relative_to(root).parts):
            continue
        if path.is_file() and not path.is_symlink():
            yield path


def scan_text(
    relpath: str,
    text: str,
    skip_cjk: bool = False,
    deny_terms: Sequence[str] = (),
) -> List[Finding]:
    """Apply the content rules to one file's text.

    Args:
        relpath: Path of the file relative to the scanned root, used in the report.
        text: Full decoded contents of the file.
        skip_cjk: If True, do not report CJK characters. An escape hatch for a local scan;
            a release must be checked without it.
        deny_terms: Extra lower-cased literal terms to report, from :func:`load_deny_terms`.

    Returns:
        The findings for this file, ordered by line number.
    """
    findings: List[Finding] = []
    check_disclosure = relpath != SELF_RELPATH
    check_cjk = not skip_cjk and relpath not in CJK_ALLOWLIST

    for lineno, line in enumerate(text.splitlines(), start=1):
        if check_disclosure:
            for category, pattern in DISCLOSURE_PATTERNS:
                for match in pattern.finditer(line):
                    findings.append(Finding(relpath, lineno, category, _truncate(match.group(0))))
            for match in SECRET_PREFIX_PATTERN.finditer(line):
                findings.append(Finding(relpath, lineno, "secret", _truncate(match.group(0))))
            for match in LONG_TOKEN_PATTERN.finditer(line):
                if looks_like_secret(match.group(0)):
                    findings.append(Finding(relpath, lineno, "secret", _truncate(match.group(0))))
        for match in EMAIL_PATTERN.finditer(line):
            if match.group(0).lower() not in ALLOWED_EMAILS:
                findings.append(Finding(relpath, lineno, "email", _truncate(match.group(0))))
        lowered = line.lower()
        for term in deny_terms:
            if term in lowered:
                findings.append(Finding(relpath, lineno, "deny-list", _truncate(term)))
        if check_cjk:
            runs = CJK_PATTERN.findall(line)
            if runs:
                findings.append(Finding(relpath, lineno, "cjk", _truncate(" ".join(runs))))
    return findings


def scan_file(
    path: Path,
    root: Path,
    skip_cjk: bool = False,
    deny_terms: Sequence[str] = (),
) -> List[Finding]:
    """Apply every rule to a single file.

    File-level rules (size, suffix, file name) are reported with line number 0. Binary files
    are exempt from the content rules only because they cannot be decoded as UTF-8.

    Args:
        path: The file to scan.
        root: Root of the scanned tree, used to build the reported relative path.
        skip_cjk: If True, do not report CJK characters.
        deny_terms: Extra lower-cased literal terms to report, from :func:`load_deny_terms`.

    Returns:
        The findings for this file.
    """
    relpath = path.relative_to(root).as_posix()
    findings: List[Finding] = []

    size = path.stat().st_size
    if size > MAX_FILE_BYTES:
        findings.append(
            Finding(relpath, 0, "oversized", f"{size / 1024 / 1024:.1f} MB > {MAX_FILE_BYTES // 1024 // 1024} MB")
        )
    if path.suffix.lower() in CHECKPOINT_SUFFIXES:
        findings.append(Finding(relpath, 0, "checkpoint", f"{path.suffix} file must not be committed"))
    if relpath != SELF_RELPATH:
        for category, pattern in DISCLOSURE_PATTERNS:
            match = pattern.search(relpath)
            if match:
                findings.append(Finding(relpath, 0, f"{category}-in-filename", _truncate(match.group(0))))
    lowered_relpath = relpath.lower()
    for term in deny_terms:
        if term in lowered_relpath:
            findings.append(Finding(relpath, 0, "deny-list-in-filename", _truncate(term)))

    try:
        text = path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return findings
    findings.extend(scan_text(relpath, text, skip_cjk=skip_cjk, deny_terms=deny_terms))
    return findings


def scan_tree(
    root: Path,
    skip_cjk: bool = False,
    deny_terms: Sequence[str] = (),
    exclude: Sequence[Path] = (),
) -> List[Finding]:
    """Scan every file in the tree.

    Args:
        root: Root of the tree to scan.
        skip_cjk: If True, do not report CJK characters.
        deny_terms: Extra lower-cased literal terms to report, from :func:`load_deny_terms`.
        exclude: Files to leave unscanned. The term file named by ``--deny-file`` goes here, so
            that pointing the scanner at a tree that also holds it cannot report it against
            itself.

    Returns:
        All findings, ordered by path and then by line number.

    Raises:
        NotADirectoryError: If ``root`` is not an existing directory.
    """
    if not root.is_dir():
        raise NotADirectoryError(f"not a directory: {root}")
    excluded = {path.resolve() for path in exclude}
    findings: List[Finding] = []
    for path in iter_files(root):
        if path.resolve() in excluded:
            continue
        findings.extend(scan_file(path, root, skip_cjk=skip_cjk, deny_terms=deny_terms))
    return findings


def summarise(findings: Sequence[Finding]) -> Dict[str, int]:
    """Count findings per category.

    Args:
        findings: The findings to summarise.

    Returns:
        A mapping from category to number of findings, ordered by descending count.
    """
    counts: Dict[str, int] = {}
    for finding in findings:
        counts[finding.category] = counts.get(finding.category, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the scanner from the command line.

    Args:
        argv: Argument vector, defaulting to ``sys.argv[1:]``.

    Returns:
        0 if the tree is clean, 1 if any finding was reported.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0], allow_abbrev=False)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parent.parent,
        help="root of the tree to scan (default: the repository root)",
    )
    parser.add_argument(
        "--skip-cjk",
        action="store_true",
        help="do not report CJK characters; a release must be checked without this flag",
    )
    parser.add_argument(
        "--deny-file",
        type=Path,
        default=None,
        help=(
            "file of extra literal terms to search for, one per line; "
            f"defaults to the file named by ${DENY_FILE_ENV_VAR}, if any"
        ),
    )
    parser.add_argument("--json", action="store_true", help="print findings as JSON instead of text")
    args = parser.parse_args(argv)

    root = args.root.resolve()
    deny_file = resolve_deny_file(root, args.deny_file)
    deny_terms = load_deny_terms(deny_file)
    findings = scan_tree(
        root,
        skip_cjk=args.skip_cjk,
        deny_terms=deny_terms,
        exclude=() if deny_file is None else (deny_file,),
    )

    if args.json:
        print(json.dumps([finding.as_dict() for finding in findings], indent=2, ensure_ascii=False))
    else:
        print(f"release check: scanning {root}")
        if deny_file is None:
            print("release check: no deny-list file in use")
        else:
            print(f"release check: {len(deny_terms)} deny-list term(s) from {deny_file}")
        if args.skip_cjk:
            print("release check: --skip-cjk is set, CJK text is NOT being reported")
        for finding in findings:
            print(finding)
        counts = summarise(findings)
        if counts:
            print("\nfindings by category:")
            for category, count in counts.items():
                print(f"  {category}: {count}")
        print()
        if findings:
            print(f"RELEASE CHECK FAILED: {len(findings)} finding(s)")
        else:
            print("RELEASE CHECK PASSED: 0 findings")

    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main())
