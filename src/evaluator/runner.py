import logging
from datetime import datetime, timedelta
from pathlib import Path

from pydantic import SecretStr

from evaluator.eval.preproc import GoldenAligner

from .common.client import (
    DumpWriterCallback,
    LLMClientConfig,
    RateLimitedLLMClient,
    RateLimiter,
    RequestBodyBuilderCallback,
)
from .data import (
    EvaluationMetadata,
    EvaluationStatus,
    ExperimentConfig,
    ModelConfig,
    TestCase,
    Workspace,
)
from .data.export import export_experiment_results, merge_experiment_results
from .eval import EvaluationCancelled, Evaluator, eval_req_body_builder
from .pipeline import PipelineExchange, SingleTurnCollector
from .pipeline.collect import rag_req_body_builder

logger = logging.getLogger(__name__)


class EvaluationRunner:
    def __init__(
        self,
        ws: Workspace,
        sys_prompt: str,
        test_cases: list[TestCase],
        is_verbose: bool,
    ):
        self._ws = ws
        self._sys_prompt = sys_prompt
        self._test_cases = test_cases
        self._is_verbose = is_verbose

    def _new_client(
        self,
        model: ModelConfig,
        api_key: SecretStr,
        dump_writer: DumpWriterCallback,
        req_body_builder: RequestBodyBuilderCallback,
    ) -> RateLimitedLLMClient:
        client_conf = LLMClientConfig(
            model=model,
            on_server_error_behavior=self._ws.global_config.on_server_error_behavior,
            on_socket_error_behavior=self._ws.global_config.on_socket_error_behavior,
            is_verbose=self._is_verbose,
        )
        limiter = RateLimiter(model.rpm)

        return RateLimitedLLMClient(
            limiter=limiter,
            conf=client_conf,
            api_key=api_key,
            dump_writer=dump_writer,
            body_builder=req_body_builder,
        )

    def _create_evaluator(self, dump_writer: DumpWriterCallback) -> Evaluator:
        global_config = self._ws.global_config

        judge_client = self._new_client(
            model=global_config.eval_model,
            api_key=self._ws.environment.eval_model_api_key,
            dump_writer=dump_writer,
            req_body_builder=eval_req_body_builder(),
        )

        return Evaluator(
            judge_client=judge_client,
            threshold=global_config.test_threshold,
            max_workers=global_config.max_concurrent_test_case_evaluations,
        )

    def _get_system_prompt(self, experiment: ExperimentConfig) -> str:
        return (
            self._sys_prompt
            if experiment.sys_prompt_override is None
            else experiment.sys_prompt_override
        )

    def _run_experiment(
        self, evaluator: Evaluator, eval_metadata: EvaluationMetadata, exp: ExperimentConfig
    ) -> None:
        logger.info("Experiment configuration:\n%s", exp.as_pretty_str())
        start_ts = datetime.now()
        global_config = self._ws.global_config

        logger.info("Experiment '%s': Starting", exp.id)
        collector = SingleTurnCollector(
            client=self._new_client(
                model=global_config.gen_model,
                api_key=self._ws.environment.gen_model_api_key,
                dump_writer=eval_metadata.write_dump,
                req_body_builder=rag_req_body_builder(exp_conf=exp),
            )
        )
        aligner = GoldenAligner(
            client=self._new_client(
                model=global_config.gen_model,
                api_key=self._ws.environment.gen_model_api_key,
                dump_writer=eval_metadata.write_dump,
                req_body_builder=rag_req_body_builder(exp_conf=exp, verbose=False),
            )
        )
        exchanges: list[PipelineExchange] = []
        sys_prompt = self._get_system_prompt(exp)
        total = len(self._test_cases)
        logger.info("Processing loaded test cases")
        for i, test_case in enumerate(self._test_cases):
            answer, chunks = collector.collect(
                sys_prompt=sys_prompt, test_case=test_case, exp_id=exp.id
            )
            aligned = aligner.format(sys_prompt=sys_prompt, test_case=test_case, exp_id=exp.id)
            exchanges.append(PipelineExchange(case=aligned, llm_response=answer, rag_chunks=chunks))
            logger.info("[%d/%d] Processed test case '%s'", i + 1, total, test_case.id)

        eval_results = evaluator.evaluate(
            exchanges=exchanges,
            exp_id=exp.id,
            eval_metadata=eval_metadata,
        )

        out_path = self._experiment_output_path(eval_metadata=eval_metadata, exp_id=exp.id)
        export_experiment_results(exp_id=exp.id, results=eval_results, output_path=out_path)

        logger.info(
            "Experiment '%s': Finished successfully; elapsed=%ss, output_path=%s",
            exp.id,
            (datetime.now() - start_ts) / timedelta(milliseconds=1000),
            out_path,
        )

    @staticmethod
    def _experiment_output_path(
        eval_metadata: EvaluationMetadata,
        exp_id: str,
    ) -> Path:
        return eval_metadata.experiments_path / exp_id / f"{exp_id}_results.xlsx"

    def _load_experiments(self, eval_metadata: EvaluationMetadata) -> list[ExperimentConfig]:
        ws_experiments = self._ws.get_experiments()

        experiments: list[ExperimentConfig] = []
        for exp_id in eval_metadata.experiments:
            exp = ws_experiments.get(exp_id)

            if exp is not None:
                experiments.append(exp)

        return experiments

    def run_evaluation(self, eval_metadata: EvaluationMetadata) -> int:
        eval_metadata.start_ts = datetime.now()
        try:
            experiments = self._load_experiments(eval_metadata)
            logger.info("Loaded experiments: %s", ", ".join(e.id for e in experiments))

            evaluator = self._create_evaluator(eval_metadata.write_dump)
            for exp in experiments:
                self._run_experiment(evaluator=evaluator, eval_metadata=eval_metadata, exp=exp)

            merge_experiment_results(
                eval_metadata=eval_metadata, tgt=eval_metadata.root_path / "evaluation_results.xlsx"  # type: ignore[operator]
            )

            eval_metadata.status = EvaluationStatus.OK
            return 0

        except (KeyboardInterrupt, EvaluationCancelled):
            logger.info("Evaluation '%s': Aborted by user", eval_metadata.id)
            eval_metadata.status = EvaluationStatus.ABORTED
            return 130

        except Exception:
            logger.exception(
                "Evaluation '%s': Aborted due to an unrecoverable error", eval_metadata.id
            )
            eval_metadata.status = EvaluationStatus.ERROR
            return 2

        finally:
            eval_metadata.end_ts = datetime.now()
            eval_metadata.finalize()

            logger.info(
                "Evaluation '%s': Finished with status [%s]",
                eval_metadata.id,
                str(eval_metadata.status),
            )
