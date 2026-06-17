from ml.evaluate import compare_models


def main():
    df = compare_models()
    if df.empty:
        print('No MLflow runs found or experiment missing.')
    else:
        print(df.to_string(index=False))


if __name__ == '__main__':
    main()
