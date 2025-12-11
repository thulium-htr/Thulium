import logging

logger = logging.getLogger(__name__)

def run_benchmark(dataset_path: str, config_path: str):
    """
    Run a full benchmark on a dataset.
    """
    logger.info(f"Starting benchmark on {dataset_path} using {config_path}")
    # Stub: iterate and compute CER/WER
    return {"cer": 0.05, "wer": 0.12}
