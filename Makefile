.PHONY: install
install:
	uv sync --no-install-package gym --no-install-package compiler-gym
	.venv/bin/pip install pip==21.0 wheel==0.38.1
	.venv/bin/pip install gym==0.21.0 compiler-gym==0.2.5
	.venv/bin/pip install stable-baselines3==1.8.0
