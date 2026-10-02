from logging import Logger


def log_err_with_raise(logger: Logger, msg: str) -> None:
    logger.error(msg)
    raise ValueError(msg)
