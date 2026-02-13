.PHONY: run run-local run-openai serve-local

run:
	uv run --env-file .env python main.py

run-local:
	uv run --env-file .env python main.py provider=local model=qwen3_4b_thinking

run-openai:
	uv run --env-file .env python main.py provider=openai model=openai

serve-local:
	uv run python -m mlx_lm server --host 127.0.0.1 --port 8080 --model lmstudio-community/Qwen3-4B-Thinking-2507-MLX-4bit
