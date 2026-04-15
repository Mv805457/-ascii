TARGET = ascii-view
PYTHON = python3

$(TARGET): main.py ascii_view/*.py
	@echo "#!/usr/bin/env python3" > $(TARGET)
	@cat main.py >> $(TARGET)
	@chmod +x $(TARGET)

run:
	$(PYTHON) main.py

install:
	pip install pillow numpy

clean:
	rm -f $(TARGET)
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true

.PHONY: clean run install
