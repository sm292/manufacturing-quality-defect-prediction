from pathlib import Path
import zipfile
import requests
import io

RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)

URL = "https://archive.ics.uci.edu/static/public/179/secom.zip"

required_files = ["secom.data", "secom_labels.data"]

try:
    print("Downloading UCI SECOM dataset...")

    response = requests.get(URL, timeout=60)
    response.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        available = archive.namelist()

        for filename in required_files:
            matches = [
                name for name in available
                if name.split("/")[-1] == filename
            ]

            if not matches:
                raise FileNotFoundError(
                    f"{filename} was not found in the downloaded archive."
                )

            content = archive.read(matches[0])
            (RAW_DIR / filename).write_bytes(content)

    print("\nDataset downloaded successfully.")
    print(f"Location: {RAW_DIR.resolve()}")

    for filename in required_files:
        print(f"  {filename}: {(RAW_DIR / filename).stat().st_size:,} bytes")

except Exception as error:
    print(f"\nDownload failed: {error}")
    print("Download the files manually from:")
    print("https://archive.ics.uci.edu/dataset/179/secom")
    raise