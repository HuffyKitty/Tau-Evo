FROM python:3.12-slim
RUN pip install --no-cache-dir swebench==5.0.2
WORKDIR /workspace
