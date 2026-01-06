PYTHON=python
PYTHONPATH?=$(CURDIR)/src
export PYTHONPATH

.PHONY: all install run clean

all: install run

install:
	$(PYTHON) -m pip install --quiet -e .

run:
	$(PYTHON) -m halving_ml.train --config src/halving_ml/config.py

clean:
	rm -rf outputs
