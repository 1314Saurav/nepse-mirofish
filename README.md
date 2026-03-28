# nepse-mirofish

Data pipeline for scraping, processing, and seeding NEPSE (Nepal Stock Exchange) data into MiroFish.

## Project Structure

```
nepse-mirofish/
  data/
    raw/          # scraped data before processing
    processed/    # cleaned, structured JSON
    seed/         # MiroFish-ready seed files
  scrapers/       # all data collection modules
  pipeline/       # ETL and transformation scripts
  db/             # database models and migrations
  scheduler/      # APScheduler jobs
  tests/          # pytest unit tests
```

## Setup

1. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate   # Linux/macOS
   .venv\Scripts\activate      # Windows
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Configure environment variables:
   ```bash
   cp .env.example .env
   # Edit .env with your actual keys
   ```

## Environment Variables

| Variable | Description |
|---|---|
| `ANTHROPIC_API_KEY` | Anthropic API key for Claude |
| `ZEP_API_KEY` | Zep memory API key |
| `NEPSE_DB_URL` | PostgreSQL connection string |
| `NEWSAPI_KEY` | NewsAPI key for financial news |
