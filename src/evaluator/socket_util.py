import logging
import platform
import subprocess
import time

logger = logging.getLogger(__name__)


def _check_ping(host: str) -> bool:
    count_flag = "-n" if platform.system().lower() == "windows" else "-c"
    try:
        subprocess.run(
            ["ping", count_flag, "1", host],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
    except (subprocess.CalledProcessError, OSError):
        return False

    return True


def ping_host(host: str, max_retries: int, retry_interval: int) -> bool:
    logger.debug("Pinging connection ...")
    total_attempts = max_retries + 1

    for attempt in range(1, total_attempts + 1):
        if _check_ping(host):
            logger.debug("Ping successful")
            return True

        logger.debug("Attempt [%d/%d]. Ping failed.", attempt, total_attempts)

        if attempt < total_attempts:
            logger.debug("Retrying in %d ...", retry_interval)
            time.sleep(retry_interval)

    return False
