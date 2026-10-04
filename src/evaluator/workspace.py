import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum, auto
from pathlib import Path
from typing import Literal
from uuid import uuid4

from dotenv import load_dotenv
from pydantic import BaseModel, Field, SecretStr, ValidationError

from .config import ExperimentConfig, GlobalConfig, parse_experiment_config, parse_global_config

logger = logging.getLogger(__name__)


@dataclass
class ApiExchangeDump:
    case_id: str
    exp_id: str
    group: Literal["pipeline", "judge"]
    is_ok: bool
    req_body: str
    res_code: int
    res_time: int
    res_body: str


@dataclass
class Environment:
    gen_model_api_key: SecretStr
    eval_model_api_key: SecretStr


class EvaluationStatus(StrEnum):
    OK = auto()
    ERROR = auto()
    ABORTED = auto()
    UNKNOWN = auto()


class EvaluationMetadata(BaseModel):
    id: str
    start_ts: datetime | None = Field(default=None)
    end_ts: datetime | None = Field(default=None)
    status: EvaluationStatus | None = Field(default=None)
    experiments: list[str] = Field(default_factory=list)
    metrics: list[str] = Field(default_factory=list)
    root_path: Path | None = Field(default=None, exclude=True)

    def init(self) -> None:
        assert self.root_path, "Root directory path for an evaluation is not set"

        self.root_path.mkdir(parents=True)
        experiments_path = self.root_path / "experiments"
        experiments_path.mkdir()
        for exp_id in self.experiments:
            dumps_path = experiments_path / exp_id / "dumps"
            (dumps_path / "pipeline").mkdir(parents=True)
            (dumps_path / "judge").mkdir()

    def finalize(self) -> None:
        with open(self.root_path / "meta.json", "w", encoding="utf-8") as f:
            f.write(self.model_dump_json(indent=4))

    def write_dump(self, dump: ApiExchangeDump) -> None:
        assert self.root_path, "Root directory path for an evaluation is not set"

        group_dir = (
            self.root_path / "experiments" / dump.exp_id / "dumps" / dump.group / dump.case_id
        )
        group_dir.mkdir(exist_ok=True)

        prefix = "ok" if dump.is_ok else "err"
        idx = sum(1 for _ in group_dir.glob(f"{prefix}_exchange_*.json")) + 1
        path = group_dir / f"{prefix}_exchange_{idx}.json"

        try:
            res_body = json.loads(dump.res_body)
        except (json.JSONDecodeError, ValueError):
            res_body = dump.res_body

        exchange = {
            "meta": {"case_id": dump.case_id, "code": dump.res_code, "time": dump.res_time},
            "request": json.loads(dump.req_body),
            "response": res_body,
        }

        with path.open("w", encoding="utf-8") as f:
            json.dump(exchange, f, indent=4, ensure_ascii=False)


class Workspace:
    def __init__(self, path: Path):
        self._root = path
        self._env: Environment | None = None
        self._global_conf: GlobalConfig | None = None
        self._experiments: dict[str, ExperimentConfig] = {}
        self._evaluation_metadata: list[EvaluationMetadata] = []

    @property
    def environment(self) -> Environment:
        assert self._env is not None
        return self._env

    @property
    def global_config(self) -> GlobalConfig:
        assert self._global_conf is not None
        return self._global_conf

    def get_evaluation_metadata(self) -> tuple[EvaluationMetadata, ...]:
        return tuple(self._evaluation_metadata)

    def get_experiments(self) -> dict[str, ExperimentConfig]:
        return self._experiments

    def new_eval_metadata(self, exp_override: list[str] | None = None) -> EvaluationMetadata:
        id = str(uuid4())
        dir_path = self._root / "evaluations" / id
        experiments = sorted(
            self._experiments.keys()
            if not exp_override
            else self._select_experiments(exp_override=exp_override)
        )
        return EvaluationMetadata(id=id, root_path=dir_path, experiments=experiments)

    def resolve_eval(self) -> None:
        self._update_validate_ws_root_path()

        self._resolve_environment()
        self._resolve_global_config()
        self._resolve_experiment_configs()

    def resolve_read(self) -> None:
        self._update_validate_ws_root_path()
        self._resolve_experiment_configs()

        eval_dir_path = self._root / "evaluations"
        if not eval_dir_path.exists():
            os.mkdir(eval_dir_path)
            return

        metadata: list[EvaluationMetadata] = []
        for eval_dir in sorted(p for p in eval_dir_path.iterdir() if p.is_dir()):
            meta_path = eval_dir / "meta.json"
            if not meta_path.exists() or not meta_path.is_file():
                metadata.append(
                    EvaluationMetadata(
                        id=eval_dir.stem,
                        status=EvaluationStatus.UNKNOWN,
                        root_path=eval_dir,
                        start_ts=datetime.min,
                        end_ts=datetime.max,
                    )
                )
                continue

            try:
                raw = meta_path.read_text(encoding="utf-8")
                meta = EvaluationMetadata.model_validate_json(raw)
                meta.root_path = meta_path.parent
            except ValidationError as e:
                logger.warning(
                    "Unable to parse evaluation '%s': invalid meta.json (%s)", eval_dir.name, e
                )
                continue
            except Exception as e:
                logger.warning(
                    "Unexpected exception occured while parsing evaluation '%s' (%s)",
                    eval_dir.name,
                    e,
                )
                continue

            metadata.append(meta)
        self._evaluation_metadata = metadata

    def _select_experiments(self, exp_override: list[str]) -> list[str]:
        if not exp_override:
            return list(self._experiments.keys())

        selected: list[str] = []
        for exp_usr in exp_override:
            if not exp_usr in self._experiments:
                raise ValueError(f"Unknown experiment ['{exp_usr}']")
            selected.append(exp_usr)

        return selected

    def _resolve_experiment_configs(self) -> None:
        exp_dir_path = self._root / "experiments"
        if not exp_dir_path.is_dir():
            os.mkdir(exp_dir_path)

        file_paths = sorted(
            p for p in exp_dir_path.iterdir() if p.is_file() and p.suffix in {".yaml", ".yml"}
        )

        if not file_paths:
            logger.warning("No experiment configs found in %s", exp_dir_path)

        configs: dict[str, ExperimentConfig] = {}
        for config_path in file_paths:
            exp = parse_experiment_config(config_path=config_path)
            configs[exp.id] = exp

        logger.info("Resolved [%s] experiments", len(configs))

        self._experiments = configs

    def _update_validate_ws_root_path(self):
        self._resolve_usr_home()  # If VOLT_HOME is set, overwrites the _root

        if not self._root.exists():
            raise RuntimeError(f"Workspace does not exist: {self._root}")

    def _resolve_global_config(self) -> None:
        config_path = self._root / "config.toml"
        self._global_conf = parse_global_config(config_path)

    def _resolve_environment(self) -> None:
        dotenv_path = self._root / ".env"
        if dotenv_path.exists() and dotenv_path.is_file():
            load_dotenv(dotenv_path)
        self._env = Workspace._resolve_env_vars()

    def _resolve_usr_home(self):
        usr_home = os.environ.get("VOLT_HOME")
        if usr_home:
            self._root = Path(usr_home).expanduser().resolve()

    @staticmethod
    def _resolve_env_vars() -> Environment:
        gen_api_key = os.getenv("OPENWEBUI_API_KEY")
        eval_api_key = os.getenv("JUDGE_API_KEY")

        if gen_api_key is None or eval_api_key is None:
            raise RuntimeError(
                "Both OPENWEBUI_API_KEY and JUDGE_API_KEY environment variables must be set"
            )

        logger.debug("Resolved environment variables")
        return Environment(
            gen_model_api_key=SecretStr(secret_value=gen_api_key),
            eval_model_api_key=SecretStr(secret_value=eval_api_key),
        )
