import asyncio
import logging
from typing import Any, TypeVar

from deepeval.models import DeepEvalBaseLLM
from pydantic import BaseModel, ValidationError

from ..client import (
    RateLimitedLLMClient,
    RequestBodyBuilderCallback,
    openai_req_body_builder,
)
from ..data import EvaluationMetadata, ModelConfig

logger = logging.getLogger(__name__)

SchemaT = TypeVar("SchemaT", bound=BaseModel)


def eval_req_body_builder() -> RequestBodyBuilderCallback:
    def builder(
        model_conf: ModelConfig,
        prompt: str,
        sys_prompt: str | None,
        schema: type[BaseModel] | None,
    ) -> dict[str, Any]:
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
    def _extract_content(resp: dict[str, Any]) -> str:
        try:
            content = resp["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            raise ValueError("LLM response does not match expected OpenAI API response schema")

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

        content = self._extract_content(resp)

        if schema is None:
            return content

        try:
            return schema.model_validate_json(content)
        except ValidationError:
            logger.error("Judge model returned JSON that does not match %s", schema.__name__)
            raise

        raise AssertionError("unreachable")

    def generate(self, prompt: str, schema: type[SchemaT] | None = None) -> str | SchemaT:
        return self._call(prompt, schema)

    async def a_generate(self, prompt: str, schema: type[SchemaT] | None = None) -> str | SchemaT:
        return await asyncio.to_thread(self._call, prompt, schema)
