import json
from typing import Any, Sequence

from src.benchmark.components import (
    BenchmarkConfig,
    build_progress_logger,
    build_statistics,
    enrich_rows,
    final_output_filename,
    load_answer_types_by_id,
    load_prediction_rows,
    load_tags_by_id,
    output_dir,
    run_judge_stage,
    run_sympy_comparison,
    sympy_output_filename,
    write_json,
)
from src.benchmark.judge_sympy_answers import LangChainJudge, load_results
from src.benchmark.sympy_compare import SympyAnswerComparator
from src.utils.console_logging import ConsoleLogger


class Benchmark:
    """Hydra-instantiable benchmark for generated prediction files."""

    def __init__(
        self,
        config: BenchmarkConfig,
        judges: Sequence[LangChainJudge] | None = None,
        logger: ConsoleLogger | None = None,
        **_: Any,
    ) -> None:
        self.config = config
        self.judges = list(judges) if judges is not None else None
        self.progress = build_progress_logger(config, logger)
        self.comparator = SympyAnswerComparator()

    def run(self) -> dict[str, Any]:
        destination = output_dir(self.config)
        destination.mkdir(parents=True, exist_ok=True)

        rows, malformed_rows_skipped = load_prediction_rows(self.config.predictions_path, self.config.limit)
        enriched_rows, enrichment_stats = enrich_rows(
            rows,
            config=self.config,
            tags_by_id=load_tags_by_id(self.config.tags_path),
            answer_types_by_id=load_answer_types_by_id(self.config.answer_types_path),
        )

        sympy_output_path = destination / sympy_output_filename(self.config)
        sympy_summary_path = destination / self.config.sympy_summary_filename
        sympy_summary = run_sympy_comparison(
            config=self.config,
            rows=enriched_rows,
            malformed_rows_skipped=malformed_rows_skipped,
            output_path=sympy_output_path,
            summary_path=sympy_summary_path,
            comparator=self.comparator,
            progress=self.progress,
        )

        final_output_path = destination / final_output_filename(self.config)
        judge_output_path = destination / self.config.judge_output_filename
        judge_summary_path = destination / "judge_summary.json"
        judge_summary = run_judge_stage(
            config=self.config,
            sympy_output_path=sympy_output_path,
            final_output_path=final_output_path,
            judge_output_path=judge_output_path,
            judge_summary_path=judge_summary_path,
            incorrect_rows=int(sympy_summary["incorrect"]),
            judges=self.judges,
            progress=self.progress,
        )

        final_rows = load_results(final_output_path)
        statistics_path = destination / self.config.statistics_filename
        statistics = build_statistics(
            config=self.config,
            output_dir=destination,
            sympy_output_path=sympy_output_path,
            sympy_summary_path=sympy_summary_path,
            judge_output_path=judge_output_path,
            judge_summary_path=judge_summary_path,
            final_output_path=final_output_path,
            statistics_path=statistics_path,
            sympy_summary=sympy_summary,
            judge_summary=judge_summary,
            enrichment_stats=enrichment_stats,
            final_rows=final_rows,
        )
        write_json(statistics_path, statistics)

        self.progress(
            "completed "
            f"rows={statistics['rows']} final_accuracy={statistics['final_accuracy']} "
            f"final_correct={statistics['final_correct']} final_incorrect={statistics['final_incorrect']}"
        )
        if self.config.print_statistics:
            print(json.dumps(statistics, indent=2, ensure_ascii=False), flush=True)
        return statistics


__all__ = ["Benchmark", "BenchmarkConfig"]
