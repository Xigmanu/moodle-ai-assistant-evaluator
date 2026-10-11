import json
import logging
from dataclasses import dataclass

from pydantic import BaseModel

from .client import (
    RateLimitedLLMClient,
    RequestBodyBuilderCallback,
    RequestContext,
    openai_req_body_builder,
)
from .data import ExperimentConfig, ModelConfig, RequestGroup, TestCase

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LLMExchange:
    case: TestCase
    llm_response: str
    rag_chunks: list[str]


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


class SingleTurnCollector:
    def __init__(self, client: RateLimitedLLMClient):
        self._client = client

    @staticmethod
    def _extract_message_content(resp: dict, case_id: str) -> str:
        try:
            content = resp["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            logger.error("Case [%s]: Unexpected completion payload format.", case_id)

        if not content:
            logger.warning("Case [%s]: Message content is empty.", case_id)

        return content

    @staticmethod
    def _parse_verbose_payload(content: str, case_id: str) -> tuple[str, list[str]]:
        payload = json.loads(content)
        retrieval = payload.get("retrieval")
        chunks = retrieval.get("final_chunks")
        answer = payload.get("final_answer")

        return str(answer).split("<br><!-- qdrant-pipeline", 1)[0], [str(chunk) for chunk in chunks]

    def collect(
        self, sys_prompt: str | None, test_cases: list[TestCase], exp_id: str
    ) -> list[LLMExchange]:
        total = len(test_cases)
        exchanges: list[LLMExchange] = []
        for i, case in enumerate(test_cases, start=1):
            case_id = case.id
            query = case.query

            ctx = RequestContext(exp_id=exp_id, case_id=case_id, group=RequestGroup.PIPELINE)
            logger.debug("Querying an answer from LLM: %s", ctx.to_string())
            raw = self._client.retrying_call(
                prompt=query, sys_prompt=sys_prompt, schema=None, ctx=ctx
            )
            content = self._extract_message_content(raw, case_id)
            answer, chunks = self._parse_verbose_payload(content, case_id)

            logger.debug(
                "Experiment '%s', case '%s': RAG returned %d chunks", exp_id, case_id, len(chunks)
            )
            exchanges.append(
                LLMExchange(
                    case=case,
                    llm_response=answer,
                    rag_chunks=chunks,
                )
            )

            logger.info(
                "Experiment '%s', case '%s': Collected an LLM response [%d/%d]",
                exp_id,
                case_id,
                i,
                total,
            )

        return exchanges
