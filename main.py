import hydra
from hydra.utils import instantiate
from openai import APIConnectionError, NotFoundError
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="configs", config_name="default")
def main(cfg: DictConfig) -> None:
    client = instantiate(cfg.provider.client)
    if cfg.model.name is None:
        raise RuntimeError(
            f"No model mapping for provider '{cfg.provider.id}' in current model config."
        )
    try:
        resp = client.chat.completions.create(
            model=cfg.model.name,
            messages=[{"role": "user", "content": cfg.prompt}],
        )
        print(resp.choices[0].message.content or "")
    except APIConnectionError as exc:
        base_url = cfg.provider.client.get("base_url", "https://api.openai.com/v1")
        raise RuntimeError(
            f"Could not connect to model endpoint at {base_url}. "
            "If using Ollama, start it and pull the model first: "
            "`ollama pull qwen3:4b`."
        ) from exc
    except NotFoundError as exc:
        raise RuntimeError(
            f"Model '{cfg.model.name}' not found on the current provider. "
            "If using Ollama, pull it first: `ollama pull qwen3:4b`."
        ) from exc


if __name__ == "__main__":
    main()
