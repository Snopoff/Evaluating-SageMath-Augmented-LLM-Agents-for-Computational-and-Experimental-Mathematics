.PHONY: chat chat-tool chat-no-tool benchmark benchmark-tool benchmark-no-tool generate-predictions generate-predictions-tool generate-predictions-no-tool

chat:
	uv run --env-file .env python main.py $(HYDRA)

chat-tool:
	uv run --env-file .env python main.py $(HYDRA)

chat-no-tool:
	uv run --env-file .env python main.py system_prompt=no-tool 'tools=[]' $(HYDRA)

benchmark:
	uv run --env-file .env python main.py --config-name benchmark $(HYDRA)

benchmark-tool:
	uv run --env-file .env python main.py --config-name benchmark $(HYDRA)

benchmark-no-tool:
	uv run --env-file .env python main.py --config-name benchmark system_prompt=no-tool 'tools=[]' $(HYDRA)

generate-predictions:
	uv run --env-file .env python main.py --config-name generate_predictions $(HYDRA)

generate-predictions-tool:
	uv run --env-file .env python main.py --config-name generate_predictions system_prompt=v4 'tools=[sage_exec]' $(HYDRA)

generate-predictions-no-tool:
	uv run --env-file .env python main.py --config-name generate_predictions system_prompt=no-tool 'tools=[]' $(HYDRA)

#serve-mlx:
#	uv run python -m mlx_lm server --host 127.0.0.1 --port 8080 --model <model_name>
