# Container entry point for regenerating the manuscript figures.
FROM python:3.11-slim
WORKDIR /artifact
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["./reproduce.sh"]
