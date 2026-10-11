FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /srv
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app app
COPY evals evals
COPY finetune finetune
COPY start.sh .
RUN chmod +x start.sh && useradd -m runner && mkdir -p data && chown -R runner /srv
USER runner
EXPOSE 8000
HEALTHCHECK CMD python -c "import urllib.request,os;urllib.request.urlopen(f'http://localhost:{os.environ.get(\"PORT\",8000)}/health')"
CMD ["./start.sh"]
