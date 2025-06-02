# flows/schedule_precompute_features.py
from prefect import flow, task
import subprocess
from core.logger.logger import logger

@task
def run_precompute(refresh: bool = False):
    logger.info(f"🧠 Prefect Task: Running precompute_features.py (refresh={refresh})")
    cmd = ["python", "core/precompute_features.py"]
    if refresh:
        cmd.append("--refresh")
    subprocess.run(cmd, check=True)

@flow(name="Precompute Features Flow")
def precompute_features_flow(refresh: bool = False):
    run_precompute(refresh)

if __name__ == "__main__":
    precompute_features_flow(refresh=True)
