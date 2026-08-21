"""
Command-line entry point for the vdaworld pipeline.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time

from rich.console import Console
from rich.table import Table

from vdaworld.config import load_config
from vdaworld.pipeline import WorldGenerationPipeline
from vdaworld.utils.printname import printname

console = Console()


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def _print_summary(
    *,
    elapsed: float,
    tool_calls: int,
    input_tokens: int,
    output_tokens: int,
    cached_tokens: int,
    input_price: float,
    output_price: float,
    cache_price: float,
    success: bool,
) -> None:
    input_cost = ((input_tokens - cached_tokens) / 1_000_000) * input_price
    cached_cost = (cached_tokens / 1_000_000) * cache_price
    output_cost = (output_tokens / 1_000_000) * output_price
    cost = input_cost + cached_cost + output_cost

    mins, secs = divmod(int(elapsed), 60)
    duration_str = f"{mins}m {secs}s" if mins else f"{secs}s"

    table = Table(title="Run Summary", show_header=True, header_style="bold cyan")
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")

    table.add_row(
        "Status", "[green]Success[/green]" if success else "[red]Failed[/red]"
    )
    table.add_row("Duration", duration_str)
    table.add_row("Tool calls (turns)", str(tool_calls))
    table.add_row("Input tokens", f"{input_tokens:,}")
    table.add_row("Output tokens", f"{output_tokens:,}")
    table.add_row("Cached tokens", f"{cached_tokens:,}")
    table.add_row("  Input cost", f"${input_cost:.4f}")
    table.add_row("  Cached cost", f"${cached_cost:.4f}")
    table.add_row("  Output cost", f"${output_cost:.4f}")
    table.add_row("Estimated cost", f"${cost:.4f}")

    console.print()
    console.print(table)


def main() -> None:
    """Parse arguments, load configuration, and run the agentic generation pipeline."""
    printname()
    _configure_logging()

    parser = argparse.ArgumentParser(
        description="VDAWorld — VLM Physical Integrator Execution Pipeline."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to a YAML configuration file (e.g. configs/conway_stage1.yaml).",
    )
    parser.add_argument(
        "--load-existing",
        action="store_true",
        help=(
            "Skip VLM code generation and load the existing simulator_gen.py from "
            "the target output directory if it exists."
        ),
    )
    args = parser.parse_args()

    logger = logging.getLogger(__name__)
    logger.info("Loading configuration from %s.", args.config)
    config = load_config(args.config)

    pipeline = WorldGenerationPipeline(config)

    t_start = time.perf_counter()
    result = pipeline.run(load_existing=args.load_existing)
    elapsed = time.perf_counter() - t_start

    spec = config.model_spec
    success = result is not None

    if success:
        logger.info(
            "Generation complete after %d tool calls. Output: %s",
            result.tool_call_count,
            result.output_dir,
        )
        _print_summary(
            elapsed=result.elapsed_seconds or elapsed,
            tool_calls=result.tool_call_count,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            cached_tokens=result.cached_tokens,
            input_price=spec.input_price,
            output_price=spec.output_price,
            cache_price=spec.cache_price,
            success=True,
        )
    else:
        logger.error("Agentic generation failed — aborting.")
        _print_summary(
            elapsed=elapsed,
            tool_calls=0,
            input_tokens=0,
            output_tokens=0,
            cached_tokens=0,
            input_price=spec.input_price,
            output_price=spec.output_price,
            cache_price=spec.cache_price,
            success=False,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
