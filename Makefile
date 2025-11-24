PYTHON_VERSION = 3.8.12
VENV_NAME     = cspyenv

.PHONY: env install clean all

env:
	@echo ">> Installing Python $(PYTHON_VERSION) (if needed)..."
	@pyenv install -s $(PYTHON_VERSION)
	@echo ">> Creating virtualenv $(VENV_NAME) (if needed)..."
	@pyenv virtualenv $(PYTHON_VERSION) $(VENV_NAME) || true
	@echo ">> Setting local pyenv version..."
	@pyenv local $(VENV_NAME)
	@echo ">> Current pyenv version:"
	@pyenv version

install:
	@echo ">> Installing Python dependencies from requirements.txt..."
	@pip install -r requirements.txt

clean:
	@rm -f */version.txt
	@rm -f .coverage
	@rm -Rf .ipynb_checkpoints */.ipynb_checkpoints
	@rm -Rf build
	@rm -Rf */__pycache__
	@rm -f */*.pyc

all: env install
