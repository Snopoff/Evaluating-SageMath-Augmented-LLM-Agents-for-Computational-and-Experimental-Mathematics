from __future__ import annotations

import sys
from pathlib import Path

from omegaconf import OmegaConf

SRC_DIR = Path(__file__).resolve().parents[2]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from llmxm2.mcp.sage_server import SageMCPService, build_fastmcp_app


def _load_mcp_config() -> dict:
    repo_root = Path(__file__).resolve().parents[3]
    cfg_path = repo_root / "configs" / "mcp" / "default.yaml"
    cfg = OmegaConf.to_container(OmegaConf.load(cfg_path), resolve=True)
    if not isinstance(cfg, dict):
        raise ValueError("MCP config must be a mapping.")
    cfg.setdefault("progress_logs", True)
    return cfg


service = SageMCPService.from_config(_load_mcp_config())
mcp = build_fastmcp_app(service)
