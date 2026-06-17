import sys
import pathlib
from unittest.mock import MagicMock, patch

# Provide lightweight mocks for optional heavy dependencies so this script
# can run in development environments without installing everything.
sys.modules.setdefault('mlflow', MagicMock())
sys.modules.setdefault('mlflow.pytorch', MagicMock())
sys.modules.setdefault('redis', MagicMock())

mock_transformers = MagicMock()
mock_transformers.BertTokenizer = MagicMock()
mock_transformers.BertTokenizer.from_pretrained = MagicMock(side_effect=Exception("tokenizer not available"))
sys.modules.setdefault('transformers', mock_transformers)
sys.modules.setdefault('torch', MagicMock())


def main():
    with patch("mlflow.pytorch.load_model"), patch("redis.Redis"):
        # Import the app by file path so `api` doesn't need to be an installed package
        import importlib.util

        api_path = pathlib.Path(__file__).resolve().parents[1] / 'api' / 'app.py'
        spec = importlib.util.spec_from_file_location("api.app", str(api_path))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        import uvicorn

        uvicorn.run(module.app, host="127.0.0.1", port=8000, log_level="warning")


if __name__ == '__main__':
    main()
