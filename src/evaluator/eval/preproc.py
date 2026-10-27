import logging

from evaluator.common.client import RateLimitedLLMClient, RequestContext
from evaluator.data import RequestGroup, TestCase

logger = logging.getLogger(__name__)


class GoldenAligner:
    def __init__(self, client: RateLimitedLLMClient):
        self._client = client

    @staticmethod
    def _build_prompt(golden: str, sample: str) -> str:
        TASK = "Formatiere den folgenden **TEXT** entsprechend dem folgenden **FORMAT** neu. Gebe NUR den formatierten Text zurück!"
        return f"# Aufgabe\n\n{TASK}\n\n## TEXT\n\n{golden}\n\n## FORMAT\n\n{sample}"

    @staticmethod
    def _extract_message_content(resp: dict, case_id: str) -> str:
        try:
            content = resp["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            logger.error("Case '%s': Unexpected completion payload format.", case_id)

        if not content:
            logger.warning("Case '%s': Message content is empty.", case_id)

        return content

    def format(self, sample: str, test_case: TestCase, exp_id: str) -> TestCase:
        prompt = self._build_prompt(golden=test_case.exp_answer, sample=sample)

        ctx = RequestContext(exp_id=exp_id, case_id=test_case.id, group=RequestGroup.ALIGN)

        raw = self._client.retrying_call(prompt=prompt, sys_prompt=None, schema=None, ctx=ctx)
        content = self._extract_message_content(raw, case_id=test_case.id)
        pruned = content.split("<br><!-- qdrant-pipeline", 1)[0]

        logger.debug(
            "Case '%s': Aligned golden format to match pipeline answer format", test_case.id
        )
        return TestCase(
            id=test_case.id,
            vl=test_case.vl,
            query=test_case.query,
            query_type=test_case.query_type,
            query_topic=test_case.query_topic,
            exp_answer=pruned,
        )
