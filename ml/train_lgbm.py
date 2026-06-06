import mlflow
import lightgbm as lgb
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, f1_score
from sklearn.pipeline import Pipeline


def train_lgbm(X_train, X_val, y_train, y_val):
    mlflow.set_experiment("finsentinel-sentiment")

    with mlflow.start_run(run_name="lightgbm_tfidf"):
        mlflow.log_params({
            "model":          "LightGBM + TF-IDF",
            "max_features":   50000,
            "ngram_range":    "(1, 2)",
            "num_leaves":     31,
            "learning_rate":  0.1,
            "n_estimators":   200
        })

        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(
                max_features=50000,
                ngram_range=(1, 2),
                sublinear_tf=True
            )),
            ('clf', lgb.LGBMClassifier(
                num_leaves=31,
                learning_rate=0.1,
                n_estimators=200,
                class_weight='balanced'
            ))
        ])

        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_val)

        f1 = f1_score(y_val, preds, average='weighted')
        mlflow.log_metric("val_f1", f1)

        print(classification_report(
            y_val, preds,
            target_names=['Negative', 'Neutral', 'Positive']
        ))

        mlflow.sklearn.log_model(pipeline, "lgbm_pipeline")

    return pipeline
