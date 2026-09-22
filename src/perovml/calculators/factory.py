"""One strict place where a configuration name becomes a calculator or an optimizer.

Every entry point -- the CLI, the batch scripts, the parallel driver -- resolves its
``calculator:`` and ``optimizer_cls:`` keys here, so that they cannot drift apart.

The rule this module exists to enforce: **an unrecognised name is an error**. Falling back
to :class:`~perovml.calculators.mock.MockCalculator` on a typo turns a screening campaign
into a directory full of well-formed, meaningless numbers, and nothing downstream can tell
the difference. The mock has to be asked for by name.

:func:`build_calculator` also returns a provenance record -- the resolved class, the model
and its hash, the task head -- which callers write next to their results, so that an energy
can always be traced back to what produced it.

Examples:
    >>> calc, provenance = build_calculator({"calculator": "mock"})
    >>> provenance["calculator"], provenance["class"]
    ('mock', 'MockCalculator')
    >>> build_calculator({"calculator": "uma_m_1p1"})
    Traceback (most recent call last):
        ...
    ValueError: unknown calculator 'uma_m_1p1'...
"""

from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

__all__ = [
    "CALCULATOR_ALIASES",
    "OPTIMIZERS",
    "build_calculator",
    "build_optimizer_cls",
    "resolve_calculator_name",
]

CALCULATOR_ALIASES: dict[str, str] = {
    "mock": "mock",
    "mockcalculator": "mock",
    "dpa3": "dpa3",
    "dpa3_omat24": "dpa3",
    "dpa3omat24": "dpa3",
    "uma": "uma",
    "fairchem": "uma",
    "fairchemcalculator": "uma",
}
"""Accepted ``calculator:`` spellings, lower-cased, mapped to the backend they select."""

OPTIMIZERS = ("LBFGS", "BFGS", "FIRE")
"""Accepted ``optimizer_cls:`` names."""

_HASH_CACHE: dict[tuple[str, int, int], str] = {}


def resolve_calculator_name(name: Any) -> str:
    """Map a configured calculator name onto a backend, or fail.

    Args:
        name: The raw ``calculator`` value from a config. Case and surrounding
            whitespace are ignored.

    Returns:
        One of ``"mock"``, ``"dpa3"`` or ``"uma"``.

    Raises:
        ValueError: If ``name`` is missing or is not an accepted spelling. The message
            lists every accepted spelling, because the usual cause is a typo.
    """
    if name is None or str(name).strip() == "":
        raise ValueError(
            "no calculator was configured. Set `calculator` to one of "
            f"{sorted(set(CALCULATOR_ALIASES))}. Use `calculator: mock` only for smoke "
            "tests -- its energies are meaningless."
        )

    key = str(name).strip().lower()
    if key not in CALCULATOR_ALIASES:
        raise ValueError(
            f"unknown calculator {str(name)!r}. Accepted names are "
            f"{sorted(set(CALCULATOR_ALIASES))}. The mock calculator is never selected "
            "implicitly: ask for it with `calculator: mock` if that is what you want."
        )
    return CALCULATOR_ALIASES[key]


def _file_fingerprint(path: str | Path) -> dict[str, Any]:
    """Describe a model file well enough to recognise it again.

    The SHA-256 is computed once per process and path; on a multi-gigabyte checkpoint the
    first call costs a few seconds, which is the price of being able to say later which
    weights produced a number.

    Args:
        path: Path to the model file.

    Returns:
        A dict with ``path``, ``bytes`` and ``sha256``. ``sha256`` is None when the path
        is not a readable file, for instance when it names a pretrained model instead.
    """
    file_path = Path(path)
    record: dict[str, Any] = {"path": str(file_path), "bytes": None, "sha256": None}
    if not file_path.is_file():
        return record

    stat = file_path.stat()
    record["bytes"] = stat.st_size
    cache_key = (str(file_path.resolve()), stat.st_size, int(stat.st_mtime))
    if cache_key in _HASH_CACHE:
        record["sha256"] = _HASH_CACHE[cache_key]
        return record

    digest = hashlib.sha256()
    with open(file_path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    _HASH_CACHE[cache_key] = digest.hexdigest()
    record["sha256"] = _HASH_CACHE[cache_key]
    return record


def build_calculator(cfg: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
    """Instantiate the ASE calculator a configuration asks for.

    Args:
        cfg: Parsed configuration. ``calculator`` selects the backend; the backend then
            reads ``dpa3_model_path`` / ``model_path``, or ``uma_model``, ``uma_task``,
            ``uma_inference`` and ``uma_device``.

    Returns:
        A ``(calculator, provenance)`` tuple. The provenance dict names the backend, the
        concrete class, the model file and its hash, and the task head, and is meant to be
        written next to the results.

    Raises:
        ValueError: If the calculator name is unknown, or if the chosen backend needs a
            model path or name that the configuration did not supply.
    """
    backend = resolve_calculator_name(cfg.get("calculator"))
    provenance: dict[str, Any] = {"calculator": backend}

    if backend == "mock":
        from perovml.calculators.mock import MockCalculator

        logger.warning("using MockCalculator: the energies it returns are not physical")
        provenance.update({"class": "MockCalculator", "model": None})
        return MockCalculator(), provenance

    if backend == "dpa3":
        from perovml.calculators.dpa3 import DPA3Omat24Calculator

        model_path = (
            cfg.get("dpa3_model_path")
            or cfg.get("model_path")
            or os.environ.get("DPA3_OMAT24_MODEL")
            or os.environ.get("DPA3_MODEL_PATH")
        )
        if not model_path:
            raise ValueError(
                "calculator is set to dpa3 but no model path was given.\n"
                "Set `dpa3_model_path` in the config, or export DPA3_OMAT24_MODEL or "
                "DPA3_MODEL_PATH. See models/README.md for how to obtain the model."
            )
        head = cfg.get("dpa3_head", "Omat24")
        provenance.update(
            {
                "class": "DPA3Omat24Calculator",
                "head": head,
                "model": _file_fingerprint(model_path),
            }
        )
        logger.info("loading DPA-3 model %s (head %s)", model_path, head)
        return DPA3Omat24Calculator(model_path=str(model_path), head=head), provenance

    from fairchem.core import FAIRChemCalculator

    uma_model = cfg.get("uma_model") or os.environ.get("UMA_MODEL")
    if not uma_model:
        raise ValueError(
            "calculator is set to uma but no model name or path was given.\n"
            "Set `uma_model` in the config, e.g. 'uma-m-1p1' or '/path/to/uma-m-1p1.pt'."
        )

    # Accept "uma-m-1p1.pt" as a way of naming the pretrained model "uma-m-1p1".
    name_or_path = str(uma_model)
    if name_or_path.endswith(".pt") and not os.path.isfile(name_or_path):
        name_or_path = os.path.basename(name_or_path)[:-3]

    task_name = cfg.get("uma_task", "oc20")
    device = cfg.get("uma_device")
    inference = cfg.get("uma_inference", "default")
    provenance.update(
        {
            "class": "FAIRChemCalculator",
            "task_name": task_name,
            "inference_settings": inference,
            "device": device,
            "model": _file_fingerprint(name_or_path) if os.path.isfile(name_or_path) else {"path": name_or_path},
        }
    )
    logger.info("loading UMA model %s (task %s, inference %s)", name_or_path, task_name, inference)
    return (
        FAIRChemCalculator.from_model_checkpoint(
            name_or_path=name_or_path,
            task_name=task_name,
            inference_settings=inference,
            device=device,
        ),
        provenance,
    )


def build_optimizer_cls(name: Any):
    """Look up an ASE optimizer class by name.

    Args:
        name: Optimizer name, one of :data:`OPTIMIZERS`. Case-insensitive.

    Returns:
        The ASE optimizer class.

    Raises:
        ValueError: If the name is unknown. Silently substituting LBFGS would change the
            relaxation protocol without saying so.
    """
    from ase.optimize import BFGS, FIRE, LBFGS

    table = {"LBFGS": LBFGS, "BFGS": BFGS, "FIRE": FIRE}
    key = str(name).strip().upper()
    if key not in table:
        raise ValueError(f"unknown optimizer {str(name)!r}. Accepted names are {list(OPTIMIZERS)}.")
    return table[key]
