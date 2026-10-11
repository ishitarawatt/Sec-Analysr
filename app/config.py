from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # SEC requires a descriptive User-Agent with contact info.
    sec_user_agent: str = "sec-analyst-demo contact@example.com"
    anthropic_api_key: str = ""
    llm_model: str = "claude-sonnet-5-5"
    # Optional OpenAI-compatible endpoint (e.g. a vLLM server serving your fine-tune).
    vllm_base_url: str = ""
    vllm_model: str = ""

    data_dir: str = "data"
    tickers: str = "AAPL,MSFT,NVDA,GOOGL,AMZN"
    filings_per_ticker: int = 2
    max_agent_steps: int = 6
    rate_limit_per_min: int = 30
    cache_ttl_seconds: int = 600


settings = Settings()
