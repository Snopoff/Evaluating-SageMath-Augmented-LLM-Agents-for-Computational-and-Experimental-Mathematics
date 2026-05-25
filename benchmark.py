from __future__ import annotations

import hydra
import hydra.utils as hu
import rootutils
from omegaconf import DictConfig

rootutils.setup_root(__file__, indicator="pyproject.toml", pythonpath=True)


@hydra.main(version_base=None, config_path="configs", config_name="benchmark")
def main(cfg: DictConfig) -> None:
    benchmark_cfg = cfg.get("benchmark", cfg)
    benchmark = hu.instantiate(benchmark_cfg)
    benchmark.run()


if __name__ == "__main__":
    main()
