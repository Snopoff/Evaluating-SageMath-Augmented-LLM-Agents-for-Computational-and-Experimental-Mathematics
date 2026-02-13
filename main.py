import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="configs", config_name="default")
def main(cfg: DictConfig) -> None:
    client = instantiate(cfg.provider.client)
    resp = client.responses.create(model=cfg.model.name, input=cfg.prompt)
    print(resp.output_text)


if __name__ == "__main__":
    main()
