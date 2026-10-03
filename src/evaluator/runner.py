import logging
from datetime import datetime

from pydantic import SecretStr

from . import config
from .client import (
    LLMClientConfig,
    RateLimitedLLMClient,
    RateLimiter,
    RequestBodyBuilderCallback,
)
from .eval import EvaluationCancelled, run_evaluation
from .export import export_eval_results, merge_experiment_results
from .judge import eval_req_body_builder
from .pipeline import collect_llm_responses, rag_req_body_builder
from .workspace import EvaluationMetadata, EvaluationStatus, ExperimentConfig, Workspace

logger = logging.getLogger(__name__)


class EvaluationRunner:
    def __init__(
        self,
        ws: Workspace,
        sys_prompt: str,
        test_cases: list[dict],
    ):
        self._ws = ws
        self._sys_prompt = sys_prompt
        self._test_cases = test_cases

    def _new_client(
        self,
        model: config.ModelConfig,
        api_key: SecretStr,
        req_body_builder: RequestBodyBuilderCallback,
    ) -> RateLimitedLLMClient:
        client_conf = LLMClientConfig(
            model=model,
            on_server_error_behavior=self._ws.global_config.on_server_error_behavior,
            on_socket_error_behavior=self._ws.global_config.on_socket_error_behavior,
        )
        limiter = RateLimiter(model.rpm)

        return RateLimitedLLMClient(
            limiter=limiter, conf=client_conf, api_key=api_key, body_builder=req_body_builder
        )

    def _run_experiment(self, eval_metadata: EvaluationMetadata, exp: ExperimentConfig) -> None:
        logger.info("Running experiment: [%s]", exp)

        config = self._ws.global_config

        gen_client = self._new_client(
            model=config.gen_model,
            api_key=self._ws.environment.gen_model_api_key,
            req_body_builder=rag_req_body_builder(exp_conf=exp),
        )
        judge_client = self._new_client(
            model=config.eval_model,
            api_key=self._ws.environment.eval_model_api_key,
            req_body_builder=eval_req_body_builder(),
        )

        logger.debug("Built gen and judge clients")

        responses = collect_llm_responses(
            client=gen_client,
            test_cases=self._test_cases,
            sys_prompt=(
                self._sys_prompt if exp.sys_prompt_override is None else exp.sys_prompt_override
            ),
            exp_id=exp.id,
            eval_metadata=eval_metadata,
        )

        eval_results = run_evaluation(
            threshold=config.test_threshold,
            judge_client=judge_client,
            exchanges=responses,
            max_workers=config.max_concurrent_test_case_evaluations,
            eval_metadata=eval_metadata,
            exp_id=exp.id,
        )

        output_path = eval_metadata.root_path / "experiments" / exp.id / f"{exp.id}_results.xlsx"
        export_eval_results(
            exp_id=exp.id,
            results=eval_results,
            output_path=output_path,
        )

        logger.info("Experiment ['%s']: Evaluation results written to '%s'", exp.id, output_path)

    def run_evaluation(self, eval_metadata: EvaluationMetadata) -> int:
        logger.info("### Starting a VOLT AI tutor evaluation ###")
        logger.info("Global configuration: %s", self._ws.global_config)

        try:
            logger.info("Running [%s] experiments", len(eval_metadata.experiments))
            eval_metadata.start_ts = datetime.now()

            ws_experiments = self._ws.get_experiments()
            for exp_id in eval_metadata.experiments:
                exp = ws_experiments.get(exp_id)
                if exp is not None:
                    self._run_experiment(eval_metadata=eval_metadata, exp=exp)

            merge_experiment_results(
                eval_metadata=eval_metadata,
                tgt=eval_metadata.root_path / "evaluation_results.xlsx",
            )
            eval_metadata.status = EvaluationStatus.OK

        except EvaluationCancelled:
            eval_metadata.status = EvaluationStatus.ABORTED
            return 130
        except Exception as e:
            logger.critical("Evaluation aborted due to an unrecoverable error (%s)", e)
            eval_metadata.status = EvaluationStatus.ERROR
            return 2
        finally:
            eval_metadata.end_ts = datetime.now()
            eval_metadata.finalize()
            logger.info(
                "VOLT AI tutor evaluation is finished with a status [%s]",
                str(eval_metadata.status).upper(),
            )

        return 0
