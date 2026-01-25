import os
from repository import Repository, RepositoryConfig

def main():
    db_url = os.getenv("SQLITE_URL", "sqlite:///app.sqlite3")
    repo = Repository(RepositoryConfig(db_url=db_url, echo=False))
    repo.create_schema()
    print(f"OK: schema created in {db_url}")

if __name__ == "__main__":
    main()
