from __future__ import annotations

from pathlib import Path

from src.benchmark.components.config import BenchmarkConfig


def output_dir(config: BenchmarkConfig) -> Path:
    if config.output_dir is not None:
        return config.output_dir
    if config.predictions_path.is_dir():
        return config.predictions_path / "output"
    return config.predictions_path.parent / "output"


def sympy_output_filename(config: BenchmarkConfig) -> str:
    if config.sympy_output_filename:
        return config.sympy_output_filename
    if config.predictions_path.is_dir():
        return "predictions.json"
    return config.predictions_path.name


def final_output_filename(config: BenchmarkConfig) -> str:
    if config.final_output_filename:
        return config.final_output_filename
    stem = config.predictions_path.name if config.predictions_path.is_dir() else config.predictions_path.stem
    return f"{stem}_final_answers.json"
