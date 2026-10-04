import argparse
import logging
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

from tabulate import tabulate

from .data import EvaluationStatus, Workspace
from .data.catalogue import load_test_cases
from .data.export import merge_experiment_results
from .runner import EvaluationRunner

logger = logging.getLogger(__name__)


def _resolve_os_default_ws_dir() -> Path:
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA") or (Path.home() / "Appdata" / "Local")
        return (Path(base) / ".volteval").resolve()

    if sys.platform == "darwin":
        return (Path.home() / "Library" / "Application Support" / "VoltEval").resolve()

    return (Path.home() / ".volteval").resolve()


def _configure_eval_logger(eval_dir_path: Path | None, is_stream: bool) -> None:
    assert eval_dir_path is not None, "root directory is None"

    handlers: list[logging.Handler] = [logging.FileHandler(eval_dir_path / "eval.log", mode="w")]
    if is_stream:
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        handlers.append(console)

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s  %(levelname)-8s---\t[%(thread)s]\t%(name)s:\t%(message)s",
        handlers=handlers,
    )

    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("urllib3.connectionpool").setLevel(logging.WARNING)


def _init_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Volt AI tutor evaluation tool")
    commands = parser.add_subparsers(dest="cmd", required=True)

    run_cmd = commands.add_parser(
        "run", help="Run an evaluation with the selected system prompt and test cases"
    )
    run_cmd.add_argument(
        "-p",
        "--prompt",
        dest="prompt",
        required=True,
        help="Path to a text file with a system prompt",
    )
    run_cmd.add_argument(
        "-i", "--input", dest="input", required=True, help="Path to an Excel file with test cases"
    )
    run_cmd.add_argument(
        "--exp-override",
        dest="exp_override",
        nargs="+",
        metavar="EXP_ID",
        help="Run only the listed experiments",
    )
    run_cmd.add_argument(
        "--stream",
        dest="is_stream",
        action="store_true",
        help="If set streams the logs to the console",
    )
    run_cmd.set_defaults(handler=_cmd_run)

    results_cmd = commands.add_parser("results", help="Inspect and manage past evaluations")
    results_sub = results_cmd.add_subparsers(dest="subcmd", required=True)

    results_list = results_sub.add_parser("list", help="List past evaluation runs")
    results_list.set_defaults(handler=_cmd_results_list)

    results_get = results_sub.add_parser("get", help="Export the final evaluation results")
    results_get.add_argument("id", help="Id of a past evaluation run (see `results list`)")
    results_get.add_argument(
        "-o", "--output", dest="output", required=True, help="Path to an output file"
    )
    results_get.set_defaults(handler=_cmd_results_get)

    results_clear = results_sub.add_parser(
        "clear", help="Clear past ABORTED, ERROR evaluation run data"
    )
    results_clear.add_argument(
        "--purge",
        action="store_true",
        dest="is_purge",
        help="Clear all past evaluation data, including the OK runs",
    )
    results_clear.set_defaults(handler=_cmd_results_clear)

    exp_cmd = commands.add_parser("experiments", help="Inspect registered experiments")
    exp_sub = exp_cmd.add_subparsers(dest="subcmd", required=True)
    exp_list = exp_sub.add_parser("list", help="List registered experiments")
    exp_list.set_defaults(handler=_cmd_experiments_list)

    return parser


def _cmd_run(args: argparse.Namespace, ws: Workspace) -> int:
    ws.resolve_eval()

    eval_metadata = ws.new_eval_metadata(exp_override=args.exp_override)
    eval_metadata.init()

    _configure_eval_logger(eval_metadata.root_path, is_stream=args.is_stream)

    prompt_path = Path(args.prompt).resolve()
    prompt = prompt_path.read_text(encoding="utf-8").strip()
    logger.debug("Resolved global system prompt for this evaluation suite")

    input_path = Path(args.input).resolve()

    logger.info(
        "Starting an evaluation suite with the following configuration:\n%s",
        ws.global_config.pretty_print(),
    )

    test_cases = load_test_cases(input_path)
    runner = EvaluationRunner(ws=ws, sys_prompt=prompt, test_cases=test_cases)

    return runner.run_evaluation(eval_metadata=eval_metadata)


def _fmt_duration(start: datetime | None, end: datetime | None) -> str:
    if start is None or end is None:
        return "N/A"
    total = int((end - start).total_seconds())
    return f"{total // 3600:02d}:{total % 3600 // 60:02d}:{total % 60:02d}"


def _cmd_results_list(args: argparse.Namespace, ws: Workspace) -> int:
    ws.resolve_read()
    metadata = sorted(ws.get_evaluation_metadata(), key=lambda x: x.end_ts)

    headers = ["ID", "STATUS", "STARTED", "FINISHED", "DURATION", "EXPERIMENTS"]
    rows = []
    for meta in metadata:
        row = [
            meta.id,
            meta.status.upper() if meta.status is not None else "UNKNOWN",
            meta.start_ts or "N/A",
            meta.end_ts or "N/A",
            _fmt_duration(start=meta.start_ts, end=meta.end_ts),
            "\n".join(meta.experiments),
        ]
        rows.append(row)
    print(
        tabulate(
            rows,
            headers=headers,
            tablefmt="presto",
            colalign=(
                "center",
                "center",
                "center",
                "center",
                "center",
            ),
        )
        if rows
        else tabulate(
            [],
            headers=headers,
            tablefmt="presto",
        )
    )

    return 0


def _cmd_results_get(args: argparse.Namespace, ws: Workspace) -> int:
    if args.output is None:
        raise ValueError("Output path was not provided")

    ws.resolve_read()
    metadata = ws.get_evaluation_metadata()

    usr_id = args.id
    usr_out = Path(args.output).resolve()

    for meta in metadata:
        if meta.id == usr_id:
            if meta.status is not EvaluationStatus.OK:
                raise ValueError("Requested evaluation didn't finish successfully")
            merge_experiment_results(eval_metadata=meta, tgt=usr_out)
            return 0

    raise ValueError(f"No evaluation exists with the id ['{usr_id}']")


def _cmd_results_clear(args: argparse.Namespace, ws: Workspace) -> int:
    ws.resolve_read()
    metadata = ws.get_evaluation_metadata()

    filtered_paths = (
        [
            m.root_path
            for m in metadata
            if m.status
            in [EvaluationStatus.ERROR, EvaluationStatus.ABORTED, EvaluationStatus.UNKNOWN, None]
        ]
        if not args.is_purge
        else [m.root_path for m in metadata]
    )
    for path in filtered_paths:
        assert path is not None, "filtered_paths is None"
        if path.exists() and path.is_dir():
            shutil.rmtree(path)
            print(f"Removed record ['{path}']")

    return 0


def _cmd_experiments_list(args: argparse.Namespace, ws: Workspace) -> int:
    ws.resolve_read()
    experiments = ws.get_experiments().values()

    headers = [
        "ID",
        "DESCRIPTION",
        "RETRIEVAL",
        "IS_CROSS_ENCODER_RERANK",
        "COLLECTIONS",
        "IS_SYS_PROMPT_OVERRIDE",
    ]
    rows = []
    for exp in experiments:
        row = [
            exp.id,
            exp.man_name,
            exp.retrieval or "N/A",
            exp.is_cross_encoder_rerank,
            exp.collections.replace(", ", "\n") if exp.collections else "N/A",
            exp.sys_prompt_override is not None,
        ]
        rows.append(row)

    print(tabulate(rows, headers=headers, tablefmt="presto"))
    return 0


def main() -> int:
    args = _init_arg_parser().parse_args()

    # Default to this ws root dir.
    # If VOLT_HOME is set then it will be overwritten during a workspace resolution (both 'read' and 'eval')
    ws = Workspace(path=_resolve_os_default_ws_dir())

    return args.handler(args, ws)
