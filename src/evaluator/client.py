import datetime
import json
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from threading import Lock, local
from typing import Literal

import requests
import urllib3
from pydantic import BaseModel, SecretStr

from .config import ModelConfig, RetryBehavior
from .logging_util import log_err_with_raise
from .socket_util import ping_host
from .workspace import ApiExchangeDump, EvaluationMetadata

logger = logging.getLogger(__name__)


@dataclass
class LLMClientConfig:
    model: ModelConfig
    on_server_error_behavior: RetryBehavior
    on_socket_error_behavior: RetryBehavior


RequestBodyBuilderCallback = Callable[[ModelConfig, str, str | None, type[BaseModel] | None], dict]


def openai_req_body_builder(
    model_conf: ModelConfig,
    prompt: str,
    sys_prompt: str | None,
    schema: type[BaseModel] | None,
) -> dict:
    messages: list[dict] = []
    if sys_prompt:
        messages.append({"role": "system", "content": sys_prompt})
    messages.append({"role": "user", "content": prompt})

    body: dict = {
        "model": model_conf.name,
        "messages": messages,
    }
    if schema is not None:
        body["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": schema.__name__,
                "schema": schema.model_json_schema(),
                "strict": True,
            },
        }

    if model_conf.temperature is not None:
        body["temperature"] = model_conf.temperature
    if model_conf.top_p is not None:
        body["top_p"] = model_conf.top_p
    if model_conf.seed is not None:
        body["seed"] = model_conf.seed
    if model_conf.max_tokens is not None:
        body["max_tokens"] = model_conf.max_tokens

    return body


class RateLimiter:
    def __init__(self, rpm: int):
        if rpm <= 0:
            log_err_with_raise(
                logger,
                "Illegal argument: Maximum requests per minute cannot be lesser or equal to zero.",
            )
        self._interval = 60.0 / rpm
        self._lock = Lock()
        self._next_allowed = 0.0

    def acquire(self):
        with self._lock:
            now = time.monotonic()
            wait = self._next_allowed - now
            if wait > 0:
                time.sleep(wait)
                now = time.monotonic()
            self._next_allowed = now + self._interval


class RateLimitedLLMClient:
    def __init__(
        self,
        limiter: RateLimiter,
        conf: LLMClientConfig,
        api_key: SecretStr,
        body_builder: RequestBodyBuilderCallback = openai_req_body_builder,
        verify_tls: bool = False,
    ):
        self._limiter = limiter
        self._conf = conf
        self._api_key = api_key
        self._body_builder = body_builder
        self._verify_tls = verify_tls
        self._local = local()

        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    @property
    def _session(self) -> requests.Session:
        session = getattr(self._local, "session", None)
        if session is None:
            session = requests.Session()
            self._local.session = session
        return session

    def load_model(self):
        return self._session

    def get_model_name(self) -> str:
        return self._conf.model.name

    def _build_headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self._api_key.get_secret_value()}",
            "Content-Type": "application/json",
        }

    def _post(
        self,
        prompt: str,
        sys_prompt: str | None,
        schema: type[BaseModel] | None,
        exp_id: str,
        case_id: str,
        group: Literal["pipeline", "judge"],
        eval_metadata: EvaluationMetadata,
    ) -> dict | None:
        self._limiter.acquire()
        body = self._body_builder(self._conf.model, prompt, sys_prompt, schema)

        resp = self._session.post(
            self._conf.model.base_url,
            headers=self._build_headers(),
            json=body,
            timeout=(10, 3000),
            verify=self._verify_tls,
        )

        eval_metadata.write_dump(
            ApiExchangeDump(
                case_id=case_id,
                exp_id=exp_id,
                group=group,
                is_ok=resp.ok,
                req_body=json.dumps(body),
                res_code=resp.status_code,
                res_time=int(resp.elapsed / datetime.timedelta(milliseconds=1)),
                res_body=resp.text,
            )
        )
        if resp.status_code in (429, 500, 502, 503, 504):
            return None
        resp.raise_for_status()

        return resp.json()

    def retrying_call(
        self,
        prompt: str,
        sys_prompt: str | None,
        schema: type[BaseModel] | None,
        exp_id: str,
        case_id: str,
        group: Literal["pipeline", "judge"],
        eval_metadata: EvaluationMetadata,
    ) -> dict:
        server_retries = self._conf.on_server_error_behavior.max_retries
        socket_retries = self._conf.on_socket_error_behavior.max_retries
        server_attempt = 0
        socket_attempt = 0

        while True:
            try:
                result = self._post(
                    prompt, sys_prompt, schema, exp_id, case_id, group, eval_metadata
                )
            except (requests.ConnectionError, requests.Timeout) as e:
                logger.warning("Network error: (%s). Attempting to reconnect ...", e)
                network_up = ping_host(
                    "google.com",
                    max_retries=socket_retries,
                    retry_interval=self._conf.on_socket_error_behavior.retry_interval,
                )
                if not network_up:
                    logger.error("Failed to reconnect.")
                    raise

                if socket_attempt >= socket_retries:
                    logger.error("Reconnected, but the request kept failing.")
                    raise

                socket_attempt += 1
                continue
            except requests.HTTPError as e:
                logger.error(
                    "LLM server error: [%s] - %s",
                    e.response.status_code,
                    e.response.text[:1000],
                )
                raise
            except Exception as e:
                logger.error("Unexpected exception %s", e)
                raise

            if result is not None:
                return result

            if server_attempt >= server_retries:
                log_err_with_raise(logger, "Exceeded maximum LLM API request attempts")

            logger.warning(
                "Retrying the request in %ss",
                self._conf.on_server_error_behavior.retry_interval,
            )
            server_attempt += 1
            time.sleep(self._conf.on_server_error_behavior.retry_interval)
