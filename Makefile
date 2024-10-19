PYTHON_VERSION := 3.10.1
VENV_DIR := ft_mlp_venv
PYENV := $(shell command -v pyenv 2> /dev/null)
PYTHON := python3

.PHONY: all
all: init
	@echo "\nTo run the main script, first activate the virtual environment:"
	@echo "source $(VENV_DIR)/bin/activate && make info"
	@echo "\nThen to run the main script, use:"
	@echo "make run"

.PHONY: init
init: create-venv install-requirements
	@echo "\nChecking virtual environment and Python version:"
	@if [ -d "$(VENV_DIR)" ]; then \
		echo "Virtual environment exists at $(VENV_DIR)"; \
		$(VENV_DIR)/bin/python --version; \
		echo "Python path: $$($(VENV_DIR)/bin/python -c "import sys; print(sys.executable)")"; \
	else \
		echo "Error: Virtual environment not found at $(VENV_DIR)"; \
	fi
	@echo "\nSetup complete. To activate the virtual environment, run:"
	@echo "source $(VENV_DIR)/bin/activate"
	@echo "\nAfter activation, you can run the main script with:"
	@echo "make run"

.PHONY: create-venv
create-venv:
	@if [ ! -d "$(VENV_DIR)" ]; then \
		echo "Creating new virtual environment..."; \
		if [ -n "$(PYENV)" ] && $(PYENV) versions | grep -q $(PYTHON_VERSION); then \
			echo "Using pyenv Python $(PYTHON_VERSION)"; \
			$(PYENV) local $(PYTHON_VERSION); \
			$(PYENV) exec python -m venv $(VENV_DIR); \
		else \
			echo "Using system Python for virtual environment."; \
			$(PYTHON) -m venv $(VENV_DIR) || { echo "Failed to create venv. Please ensure python3-venv is installed."; exit 1; }; \
		fi; \
		$(VENV_DIR)/bin/pip install --upgrade pip; \
	else \
		echo "Using existing virtual environment."; \
	fi

.PHONY: install-requirements
install-requirements:
	@if [ -f "requirements.txt" ]; then \
		$(VENV_DIR)/bin/pip install -r requirements.txt; \
	else \
		echo "requirements.txt not found. Skipping package installation."; \
	fi

.PHONY: info
info:
	@echo "Current Python version: $$(python --version 2>&1)"
	@echo "Python executable: $$(which python)"
	@echo "Active virtual environment: $${VIRTUAL_ENV:-None}"

.PHONY: run
run:
	@if [ -d "$(VENV_DIR)" ]; then \
		$(VENV_DIR)/bin/python -m srcs.main; \
	else \
		echo "Virtual environment not found. Please run 'make init' first."; \
		exit 1; \
	fi

.PHONY: notebook
notebook:
	@echo "Starting Jupyter Notebook in the virtual environment..."
	@. $(VENV_DIR)/bin/activate && \
	python -m ipykernel install --user --name=$(VENV_DIR) --display-name "Python ($(VENV_DIR))" && \
	PYTHONPATH=$(VENV_DIR)/bin/python jupyter notebook

.PHONY: fclean
fclean:
	deactivate
	rm -rf $(VENV_DIR)

.PHONY: t
t:
	pytest -v
