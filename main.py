import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="configs", config_name="default")
def main(cfg: DictConfig) -> None:
    client = instantiate(cfg.provider.client)
    # return responses ...
    # resp = client.responses.create(model=cfg.model.name, input=cfg.prompt)
    # print(resp.output_text)

    resp = client.chat.completions.create(
        model=cfg.model.name,
        messages=[{"role": cfg.model.role, "content": cfg.prompt}],
    )

    print(resp.choices[0].message.content)


if __name__ == "__main__":
    main()
