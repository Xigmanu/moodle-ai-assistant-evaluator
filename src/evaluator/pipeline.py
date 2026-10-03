import json
import logging
from dataclasses import dataclass

from pydantic import BaseModel, Field

from .catalogue import (
    ANSWER_COL,
    ID_COL,
    QUERY_COL,
    QUERY_TOPIC,
    QUERY_TYPE,
    VL_COL,
)
from .client import (
    RateLimitedLLMClient,
    RequestBodyBuilderCallback,
    openai_req_body_builder,
)
from .config import ExperimentConfig, ModelConfig
from .logging_util import log_err_with_raise
from .workspace import EvaluationMetadata

logger = logging.getLogger(__name__)


@dataclass
class LLMExchange:
    id: str
    vl: str
    topic: str
    query_type: str
    query_text: str
    golden_answer: str
    llm_response: str
    rag_chunks: list[str] = Field(default_factory=list)

    @property
    def test_case_name(self) -> str:
        return f"ID={self.id} | VL={self.vl}"


def rag_req_body_builder(exp_conf: ExperimentConfig) -> RequestBodyBuilderCallback:
    def builder(
        model_conf: ModelConfig,
        prompt: str,
        sys_prompt: str | None,
        schema: type[BaseModel] | None,
    ) -> dict:
        body = openai_req_body_builder(model_conf, prompt, sys_prompt, schema)
        body["stream"] = False
        body["collections"] = exp_conf.collections
        body["retrieval"] = exp_conf.retrieval
        body["is_cross_encoder_rerank"] = exp_conf.is_cross_encoder_rerank
        body["verbose"] = True
        return body

    return builder


def _extract_message_content(resp: dict, case_id: str) -> str:
    try:
        content = resp["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        log_err_with_raise(logger, f"Case [{case_id}]: Unexpected completion payload.")

    if not content:
        log_err_with_raise(logger, f"Case [{case_id}]: Message content is empty.")

    return content


def _parse_verbose_payload(content: str, case_id: str) -> tuple[str, list[str]]:
    try:
        payload = json.loads(content)
    except json.JSONDecodeError:
        log_err_with_raise(
            logger,
            f"Case [{case_id}]: Message content is not a valid JSON.",
        )

    if not isinstance(payload, dict):
        log_err_with_raise(
            logger,
            f"Case [{case_id}]: Expected a JSON object, got {type(payload).__name__}.",
        )

    retrieval = payload.get("retrieval")
    if not isinstance(retrieval, dict):
        log_err_with_raise(
            logger,
            f"Case [{case_id}]: Missing or malformed 'retrieval' object.",
        )

    chunks = retrieval.get("final_chunks")
    if not isinstance(chunks, list):
        log_err_with_raise(
            logger,
            f"Case [{case_id}]: Missing or malformed 'final_chunks' array.",
        )

    answer = payload.get("final_answer")
    if answer is None:
        log_err_with_raise(
            logger,
            f"Case [{case_id}]: No 'final_answer' in the pipeline payload",
        )

    return str(answer).split("<br><!-- qdrant-pipeline", 1)[0], [str(chunk) for chunk in chunks]


def collect_llm_responses(
    client: RateLimitedLLMClient,
    test_cases: list[dict],
    sys_prompt: str | None,
    exp_id: str,
    eval_metadata: EvaluationMetadata,
) -> list[LLMExchange]:
    total = len(test_cases)

    exchanges: list[LLMExchange] = []
    for position, row in enumerate(test_cases, start=1):
        case_id = row[ID_COL]
        query = row[QUERY_COL]
        logger.info("[%s/%s] Fetching LLM answer for case [%s] ...", position, total, case_id)

        raw = client.retrying_call(
            prompt=query,
            sys_prompt=sys_prompt,
            schema=None,
            exp_id=exp_id,
            case_id=case_id,
            group="pipeline",
            eval_metadata=eval_metadata,
        )
        content = _extract_message_content(raw, case_id)
        answer, chunks = _parse_verbose_payload(content, case_id)

        logger.info("Case [%s]: RAG returned [%s] chunks", case_id, len(chunks))

        exchanges.append(
            LLMExchange(
                id=case_id,
                vl=row[VL_COL],
                topic=row[QUERY_TOPIC],
                query_type=row[QUERY_TYPE],
                query_text=query,
                golden_answer=row[ANSWER_COL],
                llm_response=answer,
                rag_chunks=chunks,
            )
        )

    logger.info("Collected %s response(s).", len(exchanges))
    return exchanges

