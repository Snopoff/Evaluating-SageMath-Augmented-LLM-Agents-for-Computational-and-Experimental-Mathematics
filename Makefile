.PHONY: chat chat-tool chat-no-tool benchmark generate-predictions generate-predictions-tool generate-predictions-no-tool test test-analysis judge-sympy

chat:
	uv run --env-file .env python main.py $(HYDRA)

chat-tool:
	uv run --env-file .env python main.py $(HYDRA)

chat-no-tool:
	uv run --env-file .env python main.py system_prompt=no-tool 'tools=[]' $(HYDRA)

benchmark:
	time caffeinate -ims uv run --env-file .env python benchmark.py $(HYDRA)

generate-predictions:
	time caffeinate -ims uv run --env-file .env python main.py --config-name generate_predictions $(HYDRA)

generate-predictions-tool:
	uv run --env-file .env python main.py --config-name generate_predictions system_prompt=v4 'tools=[sage_exec]' $(HYDRA)

generate-predictions-no-tool:
	uv run --env-file .env python main.py --config-name generate_predictions system_prompt=no-tool 'tools=[]' $(HYDRA)

test:
	uv run python -m pytest $(PYTEST)

test-analysis:
	uv run --extra analysis python -m pytest $(PYTEST)

judge-sympy:
	@test -n "$(input)" || (echo "Usage: make judge-sympy input=data/results/<model>/tool/output/<file>.json"; exit 1)
	uv run --env-file .env python src/benchmark/judge_sympy_answers.py \
		--input "$(input)" \
		--judge openai=gpt-5.5 \
		--judge anthropic=claude-opus-4-7 \
		--judge google=gemini-3.5-flash \
		--resume

#serve-mlx:
#	uv run python -m mlx_lm server --host 127.0.0.1 --port 8080 --model <model_name>
