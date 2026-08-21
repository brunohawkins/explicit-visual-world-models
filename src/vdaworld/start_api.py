"""
Entry point for launching the VDAWorld API microservices.

Usage::

    .venv/bin/python -m vdaworld.start_api
    .venv/bin/python -m vdaworld.start_api --config /path/to/api_config.yaml
"""

from __future__ import annotations

import argparse
import logging
import os
import signal
import subprocess
import sys
import time

from vdaworld.utils.printname import printname

import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

# Default config path: <repo_root>/api_config.yaml, inferred from this file's location
_PACKAGE_DIR = os.path.dirname(__file__)  # src/vdaworld/
_REPO_ROOT = os.path.abspath(os.path.join(_PACKAGE_DIR, "..", ".."))
_DEFAULT_CONFIG = os.path.join(_REPO_ROOT, "api_config.yaml")


def main() -> None:
    printname()
    parser = argparse.ArgumentParser(description="Launch VDAWorld API Microservices.")
    parser.add_argument(
        "--config",
        type=str,
        default=_DEFAULT_CONFIG,
        help="Path to the API configuration YAML file (default: <repo_root>/api_config.yaml).",
    )
    args = parser.parse_args()

    if not os.path.exists(args.config):
        logging.error("Configuration file not found: %s", args.config)
        sys.exit(1)

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    processes: list[tuple[subprocess.Popen, str]] = []

    def cleanup(signum, frame) -> None:
        logging.info("Termination signal received. Shutting down all API servers...")
        for process, name in processes:
            logging.info("Terminating %s (PID: %d)...", name, process.pid)
            try:
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                logging.warning("%s did not terminate. Killing forcefully.", name)
                try:
                    os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                except ProcessLookupError:
                    pass
        logging.info("All processes terminated cleanly.")
        sys.exit(0)

    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)

    logging.info("Starting up API servers...")

    for service_name, service_cfg in config.items():
        service_type = service_cfg.get("type", "conda")

        if service_type == "vllm":
            # ---------------------------------------------------------------
            # VLLM local LLM server — launched via uv in its own project dir
            # ---------------------------------------------------------------
            vllm_path = service_cfg.get("vllm_project_path")
            if not vllm_path or not os.path.isdir(vllm_path):
                logging.error(
                    "vllm service '%s': vllm_project_path '%s' not found.",
                    service_name,
                    vllm_path,
                )
                sys.exit(1)

            model = service_cfg["model"]
            host = service_cfg.get("host", "127.0.0.1")
            port = str(service_cfg.get("port", 8000))
            cvd = str(service_cfg.get("cvd", "0"))

            server_script = os.path.join(vllm_path, "server.py")
            cmd = [
                "uv",
                "run",
                "--directory",
                vllm_path,
                "python",
                server_script,
                "--model",
                model,
                "--host",
                host,
                "--port",
                port,
            ]

            for flag, key in [
                ("--gpu-memory-utilization", "gpu_memory_utilization"),
                ("--max-model-len", "max_model_len"),
                ("--tensor-parallel-size", "tensor_parallel_size"),
                ("--dtype", "dtype"),
                ("--tool-call-parser", "tool_call_parser"),
                ("--reasoning-parser", "reasoning_parser"),
            ]:
                if key in service_cfg:
                    cmd += [flag, str(service_cfg[key])]
            if service_cfg.get("enable_auto_tool_choice"):
                cmd += ["--enable-auto-tool-choice"]
            if service_cfg.get("trust_remote_code"):
                cmd += ["--trust-remote-code"]
            if "chat_template" in service_cfg:
                # Path is relative to vllm_project_path
                tmpl = os.path.join(vllm_path, service_cfg["chat_template"])
                cmd += ["--chat-template", tmpl]

            env = os.environ.copy()
            env.pop("VIRTUAL_ENV", None)
            env.pop("VIRTUAL_ENV_PROMPT", None)
            env["CUDA_VISIBLE_DEVICES"] = cvd

            logging.info(
                "Launching VLLM service '%s' (model=%s) on %s:%s (CVD=%s)...",
                service_name,
                model,
                host,
                port,
                cvd,
            )

        elif service_type == "cosmos_transfer":
            continue

        else:
            # ---------------------------------------------------------------
            # Standard conda-based microservice (uvicorn)
            # ---------------------------------------------------------------
            env_name = service_cfg["conda_env"]
            script_path = service_cfg["server_path"]
            port = str(service_cfg.get("port", 5000))
            cvd = str(service_cfg.get("cvd", "0"))

            cmd = [
                "conda",
                "run",
                "--no-capture-output",
                "-n",
                env_name,
                "python",
                "-m",
                "uvicorn",
                script_path.replace("/", ".").replace(".py", "") + ":app",
                "--host",
                "127.0.0.1",
                "--port",
                port,
            ]

            env = os.environ.copy()

            if "VIRTUAL_ENV" in env:
                del env["VIRTUAL_ENV"]

            venv_bin = os.path.abspath(os.path.join(_REPO_ROOT, ".venv", "bin"))
            if "PATH" in env:
                env["PATH"] = os.pathsep.join(
                    p
                    for p in env["PATH"].split(os.pathsep)
                    if not p.startswith(venv_bin)
                )

            env["CUDA_VISIBLE_DEVICES"] = cvd

            logging.info(
                "Launching %s on port %s using conda env '%s'...",
                service_name,
                port,
                env_name,
            )

        p = subprocess.Popen(
            cmd, env=env, stdout=sys.stdout, stderr=sys.stderr, start_new_session=True
        )

        time.sleep(2)
        if p.poll() is not None:
            logging.error(
                "Failed to start microservice '%s'. Exit code: %d",
                service_name,
                p.returncode,
            )
            sys.exit(1)

        processes.append((p, service_name))

    logging.info("=========================================")
    logging.info(" All Microservices Online. Press Ctrl+C to Exit.")
    logging.info("=========================================")

    try:
        while True:
            for process, name in processes:
                exit_code = process.poll()
                if exit_code is not None:
                    logging.error(
                        "Microservice '%s' terminated unexpectedly with exit code %d.",
                        name,
                        exit_code,
                    )
                    cleanup(None, None)
            time.sleep(1)
    except KeyboardInterrupt:
        cleanup(None, None)
    except Exception as e:
        logging.error("Orchestrator error: %s", e)
        cleanup(None, None)


if __name__ == "__main__":
    main()
