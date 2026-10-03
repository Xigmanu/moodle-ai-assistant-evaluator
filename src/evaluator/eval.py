import logging
import threading
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass

from deepeval.metrics import (
    AnswerRelevancyMetric,
    BaseMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    FaithfulnessMetric,
    GEval,
    HallucinationMetric,
)
from deepeval.test_case import LLMTestCase, SingleTurnParams

from .client import (
    RateLimitedLLMClient,
)
from .judge import DeepEvalJudgeModel
from .pipeline import LLMExchange
from .workspace import EvaluationMetadata

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
    exchange: LLMExchange
    metrics: list[MetricResult]


class EvaluationCancelled(Exception):
    """"""


class Evaluator:
    def __init__(
        self,
        judge_client: RateLimitedLLMClient,
        threshold: float,
        max_workers: int,
        max_attempts: int = 2,
    ) -> None:
        self._judge_client = judge_client
        self._threshold = threshold
        self._max_workers = max_workers
        self._max_attempts = max_attempts

    def evaluate(
        self, exchanges: list[LLMExchange], exp_id: str, eval_metadata: EvaluationMetadata
    ) -> list[EvaluationResult]:
        logger.info("Experiment '%s': Starting evaluation", exp_id)

        cancel_event = threading.Event()
        results: dict[int, list[MetricResult | None]] = {}
        futures: dict[Future[MetricResult], tuple[int, int]] = {}

        pool = ThreadPoolExecutor(
            max_workers=self._max_workers, thread_name_prefix=f"eval-{exp_id}"
        )
        try:
            for case_i, exch in enumerate(exchanges):
                if not exch.rag_chunks:
                    logger.warning(
                        "Experiment '%s', case '%s': RAG metrics will fail due missing RAG chunks",
                        exp_id,
                        exch.id,
                    )

                judge = DeepEvalJudgeModel(
                    client=self._judge_client,
                    eval_metadata=eval_metadata,
                    exp_id=exp_id,
                    case_id=exch.id,
                )

                metrics = self._build_metrics(judge)
                test_case = self._build_test_case(exch)

                if not eval_metadata.metrics:
                    eval_metadata.metrics.extend(self._metric_name(m) for m in metrics)

                results[case_i] = [None] * len(metrics)
                for metric_i, metric in enumerate(metrics):
                    fut = pool.submit(
                        self._measure, metric, test_case, exch.id, exp_id, cancel_event
                    )
                    futures[fut] = (case_i, metric_i)
            for fut in as_completed(futures):
                case_i, metric_i = futures[fut]
                results[case_i][metric_i] = fut.result()

        except BaseException:
            logger.info(
                "Experiment '%s': Evaluation aborted. Waiting for running metrics to finish...",
                exp_id,
            )
            cancel_event.set()
            raise
        finally:
            pool.shutdown(wait=True, cancel_futures=True)

        logger.info("Experiment '%s': Evaluation finished", exp_id)
        return [
            EvaluationResult(
                index=i, exchange=exch, metrics=[m for m in results[i] if m is not None]
            )
            for i, exch in enumerate(exchanges)
        ]

    @staticmethod
    def _metric_name(metric: BaseMetric) -> str:
        return getattr(metric, "__name__", type(metric).__name__)

    @staticmethod
    def _build_test_case(exch: LLMExchange) -> LLMTestCase:
        return LLMTestCase(
            input=exch.query_text,
            actual_output=exch.llm_response,
            expected_output=exch.golden_answer,
            name=exch.test_case_name,
            retrieval_context=exch.rag_chunks,
            context=exch.rag_chunks,
        )

    def _measure(
        self,
        metric: BaseMetric,
        test_case: LLMTestCase,
        case_id: str,
        exp_id: str,
        cancel_event: threading.Event,
    ) -> MetricResult:
        name = self._metric_name(metric)
        last_err: Exception | None = None

        for attempt in range(1, self._max_attempts + 1):
            if cancel_event.is_set():
                raise EvaluationCancelled()

            try:
                metric.measure(test_case=test_case, _show_indicator=False)
            except Exception as e:
                last_err = e
                logger.warning(
                    "(%d/%d) Experiment '%s', case '%s': Metric '%s' finished exceptionally",
                    attempt,
                    self._max_attempts,
                    exp_id,
                    case_id,
                    name,
                    e,
                )
                continue

            logger.info(
                "Experiment '%s', case '%s': Successfully measured '%s': score=%s, is_success=%s",
                exp_id,
                case_id,
                name,
                "unknown" if metric.score is None else f"{metric.score:.4f}",
                bool(metric.success),
            )
            return MetricResult(
                name=name,
                score=metric.score,
                threshold=self._threshold,
                success=bool(metric.success),
                reason=metric.reason or "",
                error=None,
            )

        logger.error("Experiment '%s', case '%s': Failed to measure '%s'", exp_id, case_id, name)
        return MetricResult(
            name=name,
            score=None,
            threshold=self._threshold,
            success=False,
            reason=getattr(metric, "reason", "") or "",
            error=str(last_error),
        )

    def _build_metrics(self, model: DeepEvalJudgeModel) -> list[BaseMetric]:
        return [
            AnswerRelevancyMetric(
                threshold=self._threshold, verbose_mode=False, model=model, async_mode=False
            ),
            GEval(
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
                threshold=self._threshold,
                verbose_mode=False,
                model=model,
                async_mode=False,
            ),
            # RAG metrics
            ContextualPrecisionMetric(threshold=self._threshold, model=model, async_mode=False),
            ContextualRecallMetric(threshold=self._threshold, model=model, async_mode=False),
            FaithfulnessMetric(threshold=self._threshold, model=model, async_mode=False),
            HallucinationMetric(threshold=self._threshold, model=model, async_mode=False),
        ]


# TODO Temporary. Remove later
def run_evaluation(
    threshold: float,
    judge_client: RateLimitedLLMClient,
    exchanges: list[LLMExchange],
    max_workers: int,
    eval_metadata: EvaluationMetadata,
    exp_id: str,
) -> list[EvaluationResult]:
    return Evaluator(
        judge_client=judge_client, threshold=threshold, max_workers=max_workers
    ).evaluate(exchanges, exp_id=exp_id, eval_metadata=eval_metadata)
