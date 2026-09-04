from src.ingestion.graph import run_ingest

if __name__ == "__main__":
    count = run_ingest()
    print(f"Done. {count} chunks embedded and stored this run.")
