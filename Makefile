.PHONY: build

build:
	@if ! command -v uv >/dev/null 2>&1; then \
		echo "Installing uv..."; \
		pip install uv; \
	fi
	uv build .
