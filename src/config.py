from dotenv import load_dotenv
import os

load_dotenv()

def _get(key: str, default: str = "") -> str:
    return os.getenv(key, default)

# Strip protocol prefix if user included https:// or http:// in the hostname
_raw_host = _get("CLICKHOUSE_HOST")
CLICKHOUSE_HOST: str = _raw_host.removeprefix("https://").removeprefix("http://").rstrip("/")
CLICKHOUSE_PORT: int = int(_get("CLICKHOUSE_PORT", "8443"))
CLICKHOUSE_USER: str = _get("CLICKHOUSE_USER", "default")
CLICKHOUSE_PASSWORD: str = _get("CLICKHOUSE_PASSWORD")
CLICKHOUSE_DATABASE: str = _get("CLICKHOUSE_DATABASE", "default")


def validate_config() -> None:
    """Raise ValueError if required credentials are missing. Call before connecting."""
    missing = [k for k, v in [
        ("CLICKHOUSE_HOST", CLICKHOUSE_HOST),
        ("CLICKHOUSE_USER", CLICKHOUSE_USER),
        ("CLICKHOUSE_PASSWORD", CLICKHOUSE_PASSWORD),
    ] if not v]
    if missing:
        raise ValueError(
            f"Missing ClickHouse credentials: {', '.join(missing)}\n"
            "Copy .env.example to .env and fill in your ClickHouse Cloud details.\n"
            "Find them at: Settings → Connection details → HTTPS"
        )


if __name__ == "__main__":
    validate_config()
    print(f"host={CLICKHOUSE_HOST} port={CLICKHOUSE_PORT} db={CLICKHOUSE_DATABASE}")
