import sys
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient


# Ensure lightweight mocks are present so `patch()` doesn't try to import missing
# optional dependencies like `redis` or `mlflow` in CI/lightweight environments.
sys.modules.setdefault('mlflow', MagicMock())
sys.modules.setdefault('mlflow.pytorch', MagicMock())
sys.modules.setdefault('redis', MagicMock())
mock_transformers = MagicMock()
mock_transformers.BertTokenizer = MagicMock()
mock_transformers.BertTokenizer.from_pretrained = MagicMock(side_effect=Exception("tokenizer not available"))
sys.modules.setdefault('transformers', mock_transformers)

sys.modules.setdefault('torch', MagicMock())


def run_smoke():
    with patch("mlflow.pytorch.load_model"), patch("redis.Redis"):
        # Import the app module by file path to avoid requiring `api` to be a package
        import importlib.util
        import pathlib

        api_path = pathlib.Path(__file__).resolve().parents[1] / 'api' / 'app.py'
        spec = importlib.util.spec_from_file_location("api.app", str(api_path))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        client = TestClient(module.app)
        resp = client.get("/health")
        print(resp.status_code, resp.json())
        assert resp.status_code == 200
        assert resp.json().get("status") == "healthy"


if __name__ == '__main__':
    run_smoke()
