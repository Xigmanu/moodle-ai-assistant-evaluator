import logging
from datetime import datetime
from pathlib import Path

from pydantic import SecretStr

from .client import (
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
from .data.export import export_eval_results, merge_experiment_results
from .eval import EvaluationCancelled, Evaluator, eval_req_body_builder
from .pipeline import collect_llm_responses, rag_req_body_builder

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
            limiter=limiter, conf=client_conf, api_key=api_key, body_builder=req_body_builder
        )

    def _create_evaluator(self) -> Evaluator:
        global_config = self._ws.global_config

        judge_client = self._new_client(
            model=global_config.eval_model,
            api_key=self._ws.environment.eval_model_api_key,
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
        logger.info("Experiment configuration:\n%s", exp.pretty_print())

        global_config = self._ws.global_config
        gen_client = self._new_client(
            model=global_config.gen_model,
            api_key=self._ws.environment.gen_model_api_key,
            req_body_builder=rag_req_body_builder(exp_conf=exp),
        )

        logger.info("Experiment '%s': Starting", exp.id)
        exchanges = collect_llm_responses(
            client=gen_client,
            test_cases=self._test_cases,
            sys_prompt=self._get_system_prompt(exp),
            exp_id=exp.id,
            eval_metadata=eval_metadata,
        )

        eval_results = evaluator.evaluate(
            exchanges=exchanges,
            exp_id=exp.id,
            eval_metadata=eval_metadata,
        )

        out_path = self._experiment_output_path(eval_metadata=eval_metadata, exp_id=exp.id)
        export_eval_results(exp_id=exp.id, results=eval_results, output_path=out_path)

        logger.info(
            "Experiment '%s': Results written to '%s'",
            exp.id,
            out_path,
        )

    @staticmethod
    def _experiment_output_path(
        eval_metadata: EvaluationMetadata,
        exp_id: str,
    ) -> Path:
        return eval_metadata.root_path / "experiments" / exp_id / f"{exp_id}_results.xlsx"

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
        logger.debug(
            "Evaluation '%s': Started at (%s)", eval_metadata.id, str(eval_metadata.start_ts)
        )

        try:
            experiments = self._load_experiments(eval_metadata)
            logger.debug("Loaded '%d' experiments", len(experiments))

            evaluator = self._create_evaluator()

            for exp in experiments:
                self._run_experiment(evaluator=evaluator, eval_metadata=eval_metadata, exp=exp)

            merge_experiment_results(
                eval_metadata=eval_metadata, tgt=eval_metadata.root_path / "evaluation_results.xlsx"
            )

            eval_metadata.status = EvaluationStatus.OK
            return 0

        except EvaluationCancelled:
            logger.info("Evaluation was cancelled")
            eval_metadata.status = EvaluationStatus.ABORTED
            return 130

        except Exception:
            logger.exception("Evaluation aborted due to an unrecoverable error")
            eval_metadata.status = EvaluationStatus.ERROR
            return 2

        finally:
            eval_metadata.end_ts = datetime.now()
            eval_metadata.finalize()

            logger.info("Evaluation suite finished with status [%s]", str(eval_metadata.status))
