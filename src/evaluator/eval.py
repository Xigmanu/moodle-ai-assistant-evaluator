import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import TypeVar

from deepeval.metrics import (
    AnswerRelevancyMetric,
    BaseMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    FaithfulnessMetric,
    GEval,
    HallucinationMetric,
)
from deepeval.models import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase, SingleTurnParams
from pydantic import BaseModel, ValidationError

from .client import (
    RateLimitedLLMClient,
    RequestBodyBuilderCallback,
    openai_req_body_builder,
)
from .config import ModelConfig
from .logging_util import log_err_with_raise
from .pipeline import LLMResponse
from .workspace import EvaluationMetadata

SchemaT = TypeVar("SchemaT", bound=BaseModel)

logger = logging.getLogger(__name__)


@dataclass
class MetricResult:
    name: str
    score: float | None
    threshold: float
    success: bool
    reason: str
    error: str | None


@dataclass
class EvaluationResult:
    index: int
    response: LLMResponse
    metrics: list[MetricResult]


def eval_req_body_builder() -> RequestBodyBuilderCallback:
    def builder(
        model_conf: ModelConfig,
        prompt: str,
        sys_prompt: str | None,
        schema: type[BaseModel] | None,
    ) -> dict:
        body = openai_req_body_builder(model_conf, prompt, sys_prompt, schema)
        body["chat_template_kwargs"] = {"reasoning_strength": model_conf.reasoning_strength}
        return body

    return builder


class DeepEvalJudgeModel(DeepEvalBaseLLM):
    def __init__(
        self,
        client: RateLimitedLLMClient,
        eval_metadata: EvaluationMetadata,
        exp_id: str,
        case_id: str,
    ):
        self._client = client
        self._eval_metadata = eval_metadata
        self._exp_id = exp_id
        self._case_id = case_id

    def load_model(self):
        return self._client.load_model()

    def get_model_name(self) -> str:
        return self._client.get_model_name()

    @staticmethod
    def _extract_content(resp: dict) -> str:
        try:
            content = resp["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            log_err_with_raise(
                logger,
                "LLM response does not match expected OpenAI API response schema",
            )

        if not content:
            logger.warning("Judge model returned an empty response.")
            return ""

        return content

    def _call(self, prompt: str, schema: type[SchemaT] | None) -> str | SchemaT:
        resp = self._client.retrying_call(
            prompt=prompt,
            sys_prompt=None,
            schema=schema,
            exp_id=self._exp_id,
            case_id=self._case_id,
            group="judge",
            eval_metadata=self._eval_metadata,
        )

        text = self._extract_content(resp)

        if schema is None:
            return text

        try:
            return schema.model_validate_json(text)
        except ValidationError:
            log_err_with_raise(
                logger,
                f"Judge model returned JSON that does not match {schema.__name__}",
            )

    def generate(self, prompt: str, schema: type[SchemaT] | None = None) -> str | SchemaT:
        return self._call(prompt, schema)

    async def a_generate(self, prompt: str, schema: type[SchemaT] | None = None) -> str | SchemaT:
        return await asyncio.to_thread(self._call, prompt, schema)


def _build_test_case(resp: LLMResponse) -> LLMTestCase:
    return LLMTestCase(
        input=resp.query_text,
        actual_output=resp.llm_response,
        expected_output=resp.golden_answer,
        name=resp.test_case_name,
        retrieval_context=resp.rag_chunks,
        context=resp.rag_chunks,
    )


def _build_metrics(model: DeepEvalJudgeModel, threshold: float) -> list[BaseMetric]:
    ans_relevancy = AnswerRelevancyMetric(
        threshold=threshold, verbose_mode=False, model=model, async_mode=False
    )
    correctness = GEval(
        name="Correctness",
        criteria=(
            "Determine whether the actual output is factually correct and "
            "semantically equivalent to the expected output. "
            "Award a high score when the core facts match, even if wording differs."
        ),
        evaluation_params=[
            SingleTurnParams.INPUT,
            SingleTurnParams.ACTUAL_OUTPUT,
            SingleTurnParams.EXPECTED_OUTPUT,
        ],
        threshold=threshold,
        verbose_mode=False,
        model=model,
        async_mode=False,
    )
    ctx_precision = ContextualPrecisionMetric(threshold=threshold, model=model, async_mode=False)
    ctx_recall = ContextualRecallMetric(threshold=threshold, model=model, async_mode=False)
    faithfulness = FaithfulnessMetric(threshold=threshold, model=model, async_mode=False)
    hallucination = HallucinationMetric(threshold=threshold, model=model, async_mode=False)

    return [
        ans_relevancy,
        correctness,
        # RAG metrics
        ctx_precision,
        ctx_recall,
        faithfulness,
        hallucination,
    ]


def _metric_name(metric: BaseMetric):
    return getattr(metric, "__name__", type(metric).__name__)


def _evaluate_case(
    index: int,
    resp: LLMResponse,
    judge_client: RateLimitedLLMClient,
    exp_id: str,
    eval_metadata: EvaluationMetadata,
    threshold: float,
    cancel_event: threading.Event,
) -> EvaluationResult:
    logger.info("Experiment ['%s'] | Case [%s]: Starting case evaluation ...", exp_id, resp.id)

    judge = DeepEvalJudgeModel(
        client=judge_client, eval_metadata=eval_metadata, exp_id=exp_id, case_id=resp.id
    )

    if not resp.rag_chunks:
        logger.warning(
            "Experiment ['%s'] | Case [%s]: RAG metrics will fail, because no RAG chunks were provided",
            exp_id,
            resp.id,
        )

    test_case = _build_test_case(resp)

    attempt = 0

    metrics = _build_metrics(judge, threshold)
    if not eval_metadata.metrics:
        eval_metadata.metrics.extend([_metric_name(m) for m in metrics])

    results: list[MetricResult] = []
    for metric in _build_metrics(judge, threshold):
        name = _metric_name(metric)
        while True:
            if cancel_event is not None and cancel_event.is_set():
                logger.debug(
                    "Experiment ['%s'] | Case [%s]: Requested cancelation", exp_id, resp.id
                )
                raise KeyboardInterrupt()
            try:
                metric.measure(test_case, _show_indicator=False)
                result = MetricResult(
                    name=name,
                    score=metric.score,
                    threshold=threshold,
                    success=bool(metric.success),
                    reason=metric.reason or "",
                    error=None,
                )

                logger.info(
                    "[%s/3] Experiment ['%s'] | Case [%s]: Successfully measured [%s]: score=%s, is_success=%s",
                    attempt + 1,
                    exp_id,
                    resp.id,
                    name,
                    "unknown" if result.score is None else f"{result.score:.4f}",
                    result.success,
                )
                break
            except Exception as e:
                logger.warning(
                    "Experiment ['%s'] | Case [%s]: Metric [%s] finished exceptionally (%s)",
                    exp_id,
                    resp.id,
                    name,
                    e,
                )
                # TODO Make better
                if attempt == 1:
                    logger.error(
                        "Experiment ['%s'] | Case [%s]: Metric [%s] finished exceptionally (%s)",
                        exp_id,
                        resp.id,
                        name,
                        e,
                    )
                    result = MetricResult(
                        name=name,
                        score=None,
                        threshold=threshold,
                        success=False,
                        reason=getattr(metric, "reason", "") or "",
                        error=str(e),
                    )
                    break
                attempt += 1
        results.append(result)

    logger.info("Experiment ['%s'] | Case [%s]: Case evaluation finished", exp_id, resp.id)
    return EvaluationResult(index=index, response=resp, metrics=results)


def run_evaluation(
    threshold: float,
    judge_client: RateLimitedLLMClient,
    responses: list[LLMResponse],
    max_workers: int,
    eval_metadata: EvaluationMetadata,
    exp_id: str,
) -> list[EvaluationResult]:
    logger.info("Experiment ['%s']: Starting evaluation on [%d] worker(s) ...", exp_id, max_workers)

    cancel_event = threading.Event()
    results: list[EvaluatedResponse] = []

    if max_workers <= 1:
        for index, resp in enumerate(responses):
            if cancel_event.is_set():
                break
            results.append(
                _evaluate_case(
                    index=index,
                    resp=resp,
                    judge_client=judge_client,
                    exp_id=exp_id,
                    eval_metadata=eval_metadata,
                    threshold=threshold,
                    cancel_event=cancel_event,
                )
            )
    else:
        pool = ThreadPoolExecutor(max_workers=max_workers)
        try:
            futures = [
                pool.submit(
                    _evaluate_case,
                    index,
                    resp,
                    judge_client,
                    exp_id,
                    eval_metadata,
                    threshold,
                    cancel_event,
                )
                for index, resp in enumerate(responses)
            ]
            for fut in as_completed(futures):
                results.append(fut.result())
        except KeyboardInterrupt:
            logger.info(
                "Captured a keyboard interruption event. Waiting for worker threads to abort"
            )
            cancel_event.set()
            for fut in futures:
                fut.cancel()
            raise
        finally:
            pool.shutdown(cancel_futures=True)

    results.sort(key=lambda r: r.index)

    logger.info("Experiment ['%s']: Evaluation finished", exp_id)
    return results
