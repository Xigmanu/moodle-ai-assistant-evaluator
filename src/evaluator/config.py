import tomllib
from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass
class RetryBehavior:
    max_retries: int
    retry_interval: int


@dataclass
class ModelConfig:
    name: str
    base_url: str
    rpm: int
    temperature: float | None
    top_p: float | None
    seed: int | None
    max_tokens: int | None
    reasoning_strength: str | None


@dataclass
class GlobalConfig:
    test_threshold: float
    max_concurrent_test_case_evaluations: int
    gen_model: ModelConfig
    eval_model: ModelConfig
    on_server_error_behavior: RetryBehavior
    on_socket_error_behavior: RetryBehavior


@dataclass
class ExperimentConfig:
    id: str
    man_name: str
    retrieval: str
    is_cross_encoder_rerank: bool
    collections: str
    sys_prompt_override: str | None


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
        collections=", ".join(collections) if collections else None,
        sys_prompt_override=yml_conf.get("sys_prompt_override", None),
    )
