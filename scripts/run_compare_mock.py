import sys
import pathlib
from unittest.mock import MagicMock

# Lightweight mlflow mock so this can run in CI or machines without MLflow
mock_mlflow = MagicMock()
mock_mlflow.tracking = MagicMock()
sys.modules.setdefault('mlflow', mock_mlflow)
sys.modules.setdefault('mlflow.tracking', mock_mlflow.tracking)

# Ensure repo root is on sys.path so `import ml.evaluate` works when running
# this script from the `scripts/` folder.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from ml.evaluate import compare_models


def main():
    df = compare_models()
    if df.empty:
        print('No MLflow runs found or experiment missing (mocked).')
    else:
        print(df.to_string(index=False))


if __name__ == '__main__':
    main()
