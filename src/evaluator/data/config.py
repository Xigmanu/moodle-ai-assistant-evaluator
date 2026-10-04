import tomllib
from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class RetryBehavior:
    max_retries: int
    retry_interval: int


@dataclass(frozen=True)
class ModelConfig:
    name: str
    base_url: str
    rpm: int
    temperature: float | None
    top_p: float | None
    seed: int | None
    max_tokens: int | None
    reasoning_strength: str | None


@dataclass(frozen=True)
class GlobalConfig:
    test_threshold: float
    max_concurrent_test_case_evaluations: int
    gen_model: ModelConfig
    eval_model: ModelConfig
    on_server_error_behavior: RetryBehavior
    on_socket_error_behavior: RetryBehavior

    def pretty_print(self) -> str:
        def format_model(label: str, model: ModelConfig) -> str:
            return (
                f"{label}:\n"
                f"    NAME: '{model.name}'\n"
                f"    BASE_URL: '{model.base_url}'\n"
                f"    RPM: '{model.rpm}'\n"
                f"    TEMPERATURE: '{model.temperature}'\n"
                f"    TOP_P: '{model.top_p}'\n"
                f"    SEED: '{model.seed}'\n"
                f"    MAX_TOKENS: '{model.max_tokens}'\n"
                f"    REASONING_STRENGTH: '{model.reasoning_strength}'"
            )

        return (
            f"TEST_THRESHOLD: '{self.test_threshold}'\n"
            f"MAX_CONCURRENT_TEST_CASE_EVALUATIONS: "
            f"'{self.max_concurrent_test_case_evaluations}'\n"
            f"ON_SERVER_ERROR_BEHAVIOR: '{self.on_server_error_behavior}'\n"
            f"ON_SOCKET_ERROR_BEHAVIOR: '{self.on_socket_error_behavior}'\n"
            f"{format_model('GEN_MODEL', self.gen_model)}\n"
            f"{format_model('EVAL_MODEL', self.eval_model)}"
        )


@dataclass(frozen=True)
class ExperimentConfig:
    id: str
    man_name: str
    retrieval: str
    is_cross_encoder_rerank: bool
    collections: str
    sys_prompt_override: str | None

    def pretty_print(self) -> str:
        return f"ID: '{self.id}'\nNAME: '{self.man_name}'\nRETRIEVAL: '{self.retrieval}'\nIS_CROSS_ENCODER_RERANK: '{self.is_cross_encoder_rerank}'\nCOLLECTIONS: '{self.collections}'\nSYS_PROMPT_OVERRIDE: '{self.sys_prompt_override}'"


def _parse_model_config(section: dict) -> ModelConfig:
    return ModelConfig(
        name=section["name"],
        base_url=section["base_url"],
        rpm=section.get("rpm", 15),
        temperature=section.get("temperature"),
        top_p=section.get("top_p"),
        seed=section.get("seed"),
        max_tokens=section.get("max_tokens"),
        reasoning_strength=section.get("reasoning_strength", "low"),
    )


def parse_global_config(config_path: Path) -> GlobalConfig:
    with config_path.open("rb") as f:
        toml_conf = tomllib.load(f)

    evaluator = toml_conf["evaluator"]
    models = toml_conf["model"]

    return GlobalConfig(
        test_threshold=evaluator["test_threshold"],
        max_concurrent_test_case_evaluations=evaluator["max_concurrent_test_case_evaluations"],
        gen_model=_parse_model_config(models["gen"]),
        eval_model=_parse_model_config(models["eval"]),
        on_server_error_behavior=RetryBehavior(
            max_retries=evaluator["on_llm_request_error_max_retries"],
            retry_interval=evaluator["on_llm_request_error_retry_interval"],
        ),
        on_socket_error_behavior=RetryBehavior(
            max_retries=evaluator["on_socket_error_ping_max_retries"],
            retry_interval=evaluator["on_socket_error_ping_retry_interval"],
        ),
    )


def parse_experiment_config(config_path: Path) -> ExperimentConfig:
    with config_path.open("rb") as f:
        yml_conf = yaml.load(f, Loader=yaml.SafeLoader)

    collections = yml_conf.get("collections", None)
    return ExperimentConfig(
        id=yml_conf["id"],
        man_name=yml_conf["man_name"],
        retrieval=yml_conf.get("retrieval", ""),
        is_cross_encoder_rerank=yml_conf["is_cross_encoder_rerank"],
        collections=", ".join(collections) if collections else "",
        sys_prompt_override=yml_conf.get("sys_prompt_override", None),
    )
