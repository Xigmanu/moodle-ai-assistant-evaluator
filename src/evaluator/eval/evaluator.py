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

from ..common.client import (
    RateLimitedLLMClient,
)
from ..data import EvaluationMetadata
from ..pipeline import PipelineExchange
from .judge import DeepEvalJudgeModel

logger = logging.getLogger(__name__)

EVAL_ITER_NUM = 3


@dataclass(frozen=True)
class MetricResult:
    name: str
    score: float
    threshold: float
    success: bool
    reason: str
    error: str | None


@dataclass(frozen=True)
class EvaluationResult:
    index: int
    exchange: PipelineExchange
    metrics: list[MetricResult]


class EvaluationCancelled(Exception):
    """Raised when an evaluation is cancelled by a user."""


class Evaluator:
    def __init__(
        self,
        judge_client: RateLimitedLLMClient,
        threshold: float,
        max_workers: int,
        max_attempts: int = 2,
    ) -> None:
        if max_workers < 30:
            logger.warning(
                "Running evaluation on less than 30 worker threads. This will cause longer evaluation times"
            )

        self._judge_client = judge_client
        self._threshold = threshold
        self._max_workers = max_workers
        self._max_attempts = max_attempts

    def evaluate(
        self, exchanges: list[PipelineExchange], exp_id: str, eval_metadata: EvaluationMetadata
    ) -> list[EvaluationResult]:
        logger.info("Experiment '%s': Evaluating", exp_id)

        cancel_event = threading.Event()
        results: dict[int, list[MetricResult | None]] = {}
        futures: dict[Future[MetricResult], tuple[int, int]] = {}

        t_name_pref = f"eval-{exp_id}"
        logger.debug(
            "Threadpool conf: max_workers=%d, thread_name_prefix='%s'",
            self._max_workers,
            t_name_pref,
        )
        with ThreadPoolExecutor(
            max_workers=self._max_workers, thread_name_prefix=t_name_pref
        ) as pool:
            for case_i, exch in enumerate(exchanges):
                logger.debug("Case '%s': Building metrics", exch.case.id)
                metrics = self._build_metrics(exch, exp_id, eval_metadata)

                results[case_i] = [None] * len(metrics)

                logger.debug("Case '%s': Building DeepEval test case", exch.case.id)
                test_cases = self._build_test_cases(exch)

                for metric_i, metric in enumerate(metrics):
                    future = pool.submit(
                        self._measure_n,
                        metric,
                        test_cases,
                        case_id=exch.case.id,
                        exp_id=exp_id,
                        cancel_event=cancel_event,
                    )
                    futures[future] = (case_i, metric_i)

            try:
                for future in as_completed(futures):
                    case_i, metric_i = futures[future]
                    results[case_i][metric_i] = future.result()
            except BaseException:
                logger.info(
                    "Experiment '%s': Evaluation aborted. Waiting for running metrics to finish...",
                    exp_id,
                )
                cancel_event.set()
                for future in futures:
                    future.cancel()

                raise

        logger.info("Experiment '%s': Evaluation finished", exp_id)

        return [
            EvaluationResult(
                index=i,
                exchange=exch,
                metrics=[metric for metric in results[i] if metric is not None],
            )
            for i, exch in enumerate(exchanges)
        ]

    @staticmethod
    def _metric_name(metric: BaseMetric) -> str:
        return getattr(metric, "__name__", type(BaseMetric).__name__)

    @staticmethod
    def _build_test_cases(exch: PipelineExchange) -> list[LLMTestCase]:
        cases: list[LLMTestCase] = []
        for i in range(EVAL_ITER_NUM):
            cases.append(
                LLMTestCase(
                    input=exch.case.query,
                    actual_output=exch.llm_response[i],
                    expected_output=exch.case.exp_answer,
                    name=exch.case.test_case_name,
                    retrieval_context=list(exch.rag_chunks[i]),
                    context=exch.rag_chunks[i],
                )
            )
        return cases

    def _build_metrics(
        self, exch: PipelineExchange, exp_id: str, eval_metadata: EvaluationMetadata
    ) -> list[BaseMetric]:
        if not exch.rag_chunks:
            logger.warning(
                "Experiment '%s', case '%s': "
                "RAG metrics may fail because no RAG chunks are available",
                exp_id,
                exch.case.id,
            )

        judge = DeepEvalJudgeModel(
            client=self._judge_client,
            eval_metadata=eval_metadata,
            exp_id=exp_id,
            case_id=exch.case.id,
        )

        metrics: list[BaseMetric] = [
            AnswerRelevancyMetric(
                threshold=self._threshold, verbose_mode=False, model=judge, async_mode=False
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
                model=judge,
                async_mode=False,
            ),
            # RAG metrics
            ContextualPrecisionMetric(threshold=self._threshold, model=judge, async_mode=False),
            ContextualRecallMetric(threshold=self._threshold, model=judge, async_mode=False),
            FaithfulnessMetric(threshold=self._threshold, model=judge, async_mode=False),
            HallucinationMetric(threshold=self._threshold, model=judge, async_mode=False),
        ]

        if not eval_metadata.metrics:
            eval_metadata.metrics.extend(self._metric_name(metric) for metric in metrics)

        return metrics

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
                    "[%d/%d] Case '%s': Metric '%s' finished exceptionally",
                    attempt,
                    self._max_attempts,
                    case_id,
                    name,
                )
                continue

            logger.info(
                "Case '%s': Successfully measured '%s': score=%s, is_success=%s",
                case_id,
                name,
                "unknown" if metric.score is None else f"{metric.score:.4f}",
                bool(metric.success),
            )
            return MetricResult(
                name=name,
                score=metric.score if metric.score is not None else 0.0,
                threshold=self._threshold,
                success=bool(metric.success),
                reason=metric.reason or "",
                error=None,
            )

        logger.error("Case '%s': Failed to measure '%s'", case_id, name)
        return MetricResult(
            name=name,
            score=0.0,
            threshold=self._threshold,
            success=False,
            reason=getattr(metric, "reason", "") or "",
            error=str(last_err),
        )

    def _measure_n(
        self,
        metric: BaseMetric,
        test_cases: list[LLMTestCase],
        case_id: str,
        exp_id: str,
        cancel_event: threading.Event,
    ) -> MetricResult:
        results = [self._measure(metric, tc, case_id, exp_id, cancel_event) for tc in test_cases]
        score_mean = 0.0
        combined_reason: list[str] = []
        combined_errors: list[str] = []
        for r in results:
            score_mean += r.score
            combined_reason.append(r.reason)
            combined_errors.append(r.error if r.error is not None else "NONE")
        score_mean = score_mean / EVAL_ITER_NUM

        assert metric.threshold is not None
        return MetricResult(
            name=self._metric_name(metric),
            score=score_mean,
            threshold=metric.threshold,
            success=score_mean >= metric.threshold,
            reason="|".join(combined_reason),
            error="|".join(combined_errors),
        )
