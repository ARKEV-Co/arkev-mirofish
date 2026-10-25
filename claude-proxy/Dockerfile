# Claude subscription proxy for MiroFish (see proxy.py). The official Claude Code CLI, pinned.
FROM node:22-bookworm-slim
RUN apt-get update && apt-get install -y --no-install-recommends python3 ca-certificates \
 && rm -rf /var/lib/apt/lists/* \
 && npm install -g @anthropic-ai/claude-code@2.1.296 \
 && useradd --uid 10021 --create-home --shell /usr/sbin/nologin claudeproxy
COPY proxy.py /app/proxy.py
USER claudeproxy
ENV HOME=/home/claudeproxy PROXY_PORT=8082
EXPOSE 8082
HEALTHCHECK --interval=30s --timeout=5s CMD python3 -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8082/health', timeout=4)"
CMD ["python3", "/app/proxy.py"]
