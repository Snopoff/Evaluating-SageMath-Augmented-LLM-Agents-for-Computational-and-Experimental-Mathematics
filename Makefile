.PHONY: run

run:
	uv run --env-file .env python main.py $(HYDRA)

#serve-mlx:
#	uv run python -m mlx_lm server --host 127.0.0.1 --port 8080 --model <model_name>
