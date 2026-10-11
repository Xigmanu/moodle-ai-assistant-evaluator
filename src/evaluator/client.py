import datetime
import json
import logging
import platform
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass
from threading import Lock, local
from typing import Any

import requests
import urllib3
from pydantic import BaseModel, SecretStr

from .data import ModelConfig, RequestGroup, RetryBehavior
from .data.workspace import ApiExchangeDump

logger = logging.getLogger(__name__)


DumpWriterCallback = Callable[[ApiExchangeDump], None]
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

    hyperparams = {
        "temperature": model_conf.temperature,
        "top_p": model_conf.top_p,
        "seed": model_conf.seed,
        "max_tokens": model_conf.max_tokens,
    }

    body.update({key: value for key, value in hyperparams.items() if value is not None})

    return body


@dataclass(frozen=True)
class LLMClientConfig:
    model: ModelConfig
    on_server_error_behavior: RetryBehavior
    on_socket_error_behavior: RetryBehavior
    is_verbose: bool


@dataclass
class RequestContext:
    exp_id: str
    case_id: str
    group: RequestGroup

    def to_string(self) -> str:
        return f"[e='{self.exp_id}', c='{self.case_id}', g='{self.group}']"


class RateLimiter:
    def __init__(self, rpm: int):
        if rpm <= 0:
            raise ValueError(
                "Failed to initialize rate limiter. Requests per minute must be greater than zero"
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
    _REQ_TIMEOUT = (30, 3000)
    _RETRYABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 504})

    def __init__(
        self,
        limiter: RateLimiter,
        conf: LLMClientConfig,
        api_key: SecretStr,
        dump_writer: DumpWriterCallback,
        body_builder: RequestBodyBuilderCallback = openai_req_body_builder,
        verify_tls: bool = False,
    ):
        self._limiter = limiter
        self._conf = conf
        self._api_key = api_key
        self._write_dump = dump_writer
        self._body_builder = body_builder
        self._verify_tls = verify_tls
        self._local = local()

        if not verify_tls:
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

    def _build_headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._api_key.get_secret_value()}",
            "Content-Type": "application/json",
        }

    def _write_exchange_dump(
        self, body: dict[str, Any], res: requests.Response, ctx: RequestContext
    ) -> None:
        self._write_dump(
            ApiExchangeDump(
                case_id=ctx.case_id,
                exp_id=ctx.exp_id,
                group=ctx.group,
                is_ok=res.ok,
                req_body=json.dumps(body),
                res_code=res.status_code,
                res_time=int(res.elapsed / datetime.timedelta(milliseconds=1)),
                res_body=res.text,
            )
        )

    def _post(
        self,
        prompt: str,
        sys_prompt: str | None,
        schema: type[BaseModel] | None,
        ctx: RequestContext,
    ) -> dict | None:
        self._limiter.acquire()
        body = self._body_builder(self._conf.model, prompt, sys_prompt, schema)

        res = self._session.post(
            self._conf.model.base_url,
            headers=self._build_headers(),
            json=body,
            timeout=self._REQ_TIMEOUT,
            verify=self._verify_tls,
        )

        logger.debug(
            "HTTP POST [%d] '%s' %d | %s",
            res.status_code,
            self._conf.model.base_url,
            res.elapsed / datetime.timedelta(milliseconds=1),
            ctx.to_string(),
        )
        if self._conf.is_verbose:
            self._write_exchange_dump(body=body, res=res, ctx=ctx)
        if res.status_code in self._RETRYABLE_STATUS_CODES:
            logger.debug("Request to the model API failed but returned a recoverable status code")
            return None

        res.raise_for_status()

        return res.json()

    def _check_ping(self) -> bool:
        count_flag = "-n" if platform.system().lower() == "windows" else "-c"
        try:
            subprocess.run(
                ["ping", count_flag, "1", self._conf.model.base_url],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=True,
            )
        except (subprocess.CalledProcessError, OSError):
            return False

        return True

    def _ping_host(self) -> bool:
        logger.debug("Pinging connection ...")
        socket_behavior = self._conf.on_socket_error_behavior

        total_attempts = socket_behavior.max_retries + 1

        for attempt in range(1, total_attempts + 1):
            if self._check_ping():
                logger.debug("Ping successful")
                return True

            logger.debug("Attempt [%d/%d]. Ping failed.", attempt, total_attempts)

            if attempt < total_attempts:
                logger.debug("Retrying in %d ...", socket_behavior.retry_interval)
                time.sleep(socket_behavior.retry_interval)

        return False

    def retrying_call(
        self,
        prompt: str,
        sys_prompt: str | None,
        schema: type[BaseModel] | None,
        ctx: RequestContext,
    ) -> dict:
        server_behavior = self._conf.on_server_error_behavior
        socket_behavior = self._conf.on_socket_error_behavior

        server_attempt = 0
        socket_attempt = 0

        while True:
            try:
                res = self._post(prompt, sys_prompt, schema, ctx)
            except (requests.ConnectionError, requests.Timeout) as e:
                if socket_attempt >= socket_behavior.max_retries:
                    logger.error("Network request failed")
                    raise

                socket_attempt += 1

                logger.warning("Network error: %s", e)

                if not self._ping_host():
                    logger.error("Failed to restore connection to network")
                    raise

                time.sleep(socket_behavior.retry_interval)
                continue
            except requests.HTTPError as e:
                logger.error(
                    "LLM server returned HTTP %s: %s",
                    e.response.status_code if e.response is not None else "unknown",
                    e.response.text[:1000] if e.response is not None else "",
                )
                raise
            except Exception as e:
                logger.error("Unexpected exception while calling LLM API")
                raise

            if res is not None:
                return res

            if server_attempt >= server_behavior.max_retries:
                raise ValueError("Exceeded maximum LLM API request attempts")

            server_attempt += 1

            logger.warning(
                "Retrying LLM request in %ss",
                server_behavior.retry_interval,
            )
            time.sleep(server_behavior.retry_interval)
