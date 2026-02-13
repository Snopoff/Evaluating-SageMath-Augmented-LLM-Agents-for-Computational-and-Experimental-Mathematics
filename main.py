import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="configs", config_name="default")
def main(cfg: DictConfig) -> None:
    client = instantiate(cfg.provider.client)
    resp = client.chat.completions.create(
        model=cfg.model.name,
        messages=[{"role": "user", "content": cfg.prompt}],
    )
    print(resp.choices[0].message.content or "")


if __name__ == "__main__":
    main()
