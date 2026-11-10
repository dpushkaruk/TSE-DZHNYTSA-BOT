FROM apache/airflow:3.3.1-python3.12
RUN pip install --no-cache-dir "pandas>=2.2" "pyarrow>=17" psycopg2-binary==2.9.11 pyyaml telethon==1.40.0 python-dotenv==1.1.1
