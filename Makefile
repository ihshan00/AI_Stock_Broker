PYTHON=./venv/bin/python3
PIP=./venv/bin/pip

# Define directories
DIRS=app/config app/src

.PHONY: help
help:
	@echo "setup - Set up the virtual environment, install dependencies, and create directories."
	@echo "clean - Remove the virtual environment and created directories."

venv:
	if [ ! -d "venv" ]; then \
		python3 -m venv venv; \
		$(PYTHON) -m pip install --upgrade pip; \
	fi

.PHONY: create_dirs
create_dirs:
	mkdir -p $(DIRS)

.PHONY: setup
setup: venv 
	$(PIP) install -r requirements.txt

.PHONY: clean
clean:
	rm -rf venv
