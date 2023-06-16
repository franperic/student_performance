import time
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold
import optuna
from optuna.samplers import TPESampler

from xgboost import XGBClassifier


df = pd.read_csv("data/train.csv")
labels = pd.read_csv("data/train_labels.csv")
labels = labels.assign(
    session=lambda x: x.session_id.str.split("_", expand=True)[0].astype(int),
    question=(
        lambda x: x.session_id.str.split("_", expand=True)[1]
        .str.replace("q", "")
        .astype(int)
    ),
)


# define relevant columns
CATEGORICAL = ["event_name", "name", "fqid", "room_fqid", "text_fqid"]
NUMERICAL = [
    "elapsed_time",
    "level",
    "page",
    "room_coor_x",
    "room_coor_y",
    "screen_coor_x",
    "screen_coor_y",
    "hover_duration",
]
EVENTS = [
    "navigate_click",
    "person_click",
    "cutscene_click",
    "object_click",
    "map_hover",
    "notification_click",
    "map_click",
    "observation_click",
    "checkpoint",
]


def aggregate_features(df: pd.DataFrame, agg_fns: list, cols: list) -> pd.DataFrame:
    """Aggregate features by session_id and level_group"""
    return df.groupby(["session_id", "level_group"])[cols].agg(agg_fns)


def flatten_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten multi-index columns"""
    df.columns = ["_".join(col).strip() for col in df.columns.values]
    return df.reset_index()


def feature_prep(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare features for prediction"""
    df_num = df.pipe(
        aggregate_features, agg_fns=["mean", "std", "min", "max"], cols=NUMERICAL
    ).pipe(flatten_cols)
    df_cat = df.pipe(
        aggregate_features, agg_fns=["nunique", "count"], cols=CATEGORICAL
    ).pipe(flatten_cols)
    df = pd.get_dummies(df, columns=["event_name"], prefix="", prefix_sep="")
    df_events = df.pipe(
        aggregate_features, agg_fns=["sum"], cols=EVENTS + ["elapsed_time"]
    ).pipe(flatten_cols)

    df = df_num.merge(df_cat, on=["session_id", "level_group"])
    df = df.merge(df_events, on=["session_id", "level_group"])
    return df


# split data into train and validation sets
sessions = df.session_id.unique()
# np.random.seed(42)
# test_sessions = np.random.choice(sessions, size=int(len(sessions) * 0.2), replace=False)
# train_sessions = np.setdiff1d(sessions, test_sessions)


# train_X = df.loc[df.session_id.isin(train_sessions)]
# train_y = labels.loc[labels.session.isin(train_sessions)]
# test_X = df.loc[df.session_id.isin(test_sessions)]
# test_y = labels.loc[labels.session.isin(test_sessions)]


# train_X = feature_prep(train_X)
# test_X = feature_prep(test_X)
train_X = feature_prep(df)
train_y = labels

gkf = GroupKFold(n_splits=5)
oof = pd.DataFrame(data=np.zeros((len(sessions), 18)))
oof.index = train_X.session_id.unique()

models = {}
start_time = time.time()
for fold, (train_idx, val_idx) in enumerate(
    gkf.split(train_X, groups=train_X.session_id)
):
    print(f"Fold: {fold}")
    for q in range(1, 19):
        if q <= 3:
            grp = "0-4"
        elif q <= 13:
            grp = "5-12"
        elif q <= 18:
            grp = "13-22"

        train_X_q = train_X.iloc[train_idx].pipe(lambda x: x.loc[x.level_group == grp])
        train_y_q = train_y.loc[
            (train_y.question == q)
            & train_y.session.isin(train_X_q.session_id.unique())
        ]

        val_X_q = train_X.iloc[val_idx].pipe(lambda x: x.loc[x.level_group == grp])
        val_y_q = train_y.loc[
            (train_y.question == q) & train_y.session.isin(val_X_q.session_id.unique())
        ]

        # def get_hyperparams(train_X, train_y, val_X, val_y):
        # def objective(trial):
        #     xgb_params = {
        #         "objective": "binary:logistic",
        #         "eval_metric": "logloss",
        #         "learning_rate": trial.suggest_float(
        #             "learning_rate", 0.01, 0.08
        #         ),  # tune 0.05
        #         "max_depth": trial.suggest_int("max_depth", 0, 8),  # tune 4
        #         "n_estimators": 1000,
        #         "early_stopping_rounds": 50,
        #         "tree_method": "hist",
        #         "subsample": 0.8,
        #         "colsample_bytree": 0.4,
        #         "min_child_weight": trial.suggest_float(
        #             "min_child_weight", 0.01, 0.99
        #         ),
        #     }
        #     cutoff = trial.suggest_float("cutoff", 0.01, 0.99)

        #     model = XGBClassifier(**xgb_params)
        #     model.fit(
        #         train_X.drop(["session_id", "level_group"], axis=1).astype(
        #             "float32"
        #         ),
        #         train_y,
        #         eval_set=[
        #             (
        #                 val_X.drop(["session_id", "level_group"], axis=1).astype(
        #                     "float32"
        #                 ),
        #                 val_y,
        #             )
        #         ],
        #         verbose=False,
        #     )
        #     preds = model.predict_proba(
        #         val_X.drop(["session_id", "level_group"], axis=1).astype("float32")
        #     )[:, 1]
        #     preds = (preds > cutoff).astype(int)
        #     return f1_score(val_y, preds, average="macro")

        # study = optuna.create_study(
        #     direction="maximize", sampler=TPESampler(seed=42)
        # )
        # study.optimize(objective, n_trials=100)

        # return study.best_params

        # best_params_q = get_hyperparams(
        #     train_X_q, train_y_q["correct"], val_X_q, val_y_q["correct"]
        # )

        # best_params[(fold, q)] = best_params_q.copy()
        # cutoffs[(fold, q)] = best_params_q["cutoff"]
        # best_params_q.pop("cutoff")
        xgb_params = {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "learning_rate": 0.05,
            "max_depth": 4,
            "n_estimators": 1000,
            "early_stopping_rounds": 50,
            "tree_method": "hist",
            "subsample": 0.8,
            "colsample_bytree": 0.4,
            # "min_child_weight": trial.suggest_float(
            #     "min_child_weight", 0.01, 0.99
            # ),
        }
        clf = XGBClassifier(**xgb_params)
        clf.fit(
            train_X_q.drop(["session_id", "level_group"], axis=1).astype("float32"),
            train_y_q["correct"],
            eval_set=[
                (
                    val_X_q.drop(["session_id", "level_group"], axis=1).astype(
                        "float32"
                    ),
                    val_y_q["correct"],
                )
            ],
            verbose=False,
        )
        pred = clf.predict_proba(
            val_X_q.drop(["session_id", "level_group"], axis=1).astype("float32")
        )[:, 1]
        oof.loc[val_X_q.session_id.unique(), q - 1] = clf.predict_proba(
            val_X_q.drop(["session_id", "level_group"], axis=1).astype("float32")
        )[:, 1]
        models[q] = clf
print(time.time() - start_time)


actuals = oof.copy()
for q in range(1, 19):
    tmp = labels.loc[labels.question == q].set_index("session").loc[oof.index.values]
    actuals[q - 1] = tmp.correct.values
y = np.concatenate(actuals.values, axis=0)


def objective(trial):
    # cutoffs = {}
    # for q in range(1, 19):
    #     cutoffs[q] = trial.suggest_float(f"cutoff_{q}", 0.1, 0.9)
    cutoff = trial.suggest_float(f"cutoff", 0.1, 0.9)
    pred = (oof > cutoff).astype(int)
    pred = np.concatenate(pred.values, axis=0)

    return f1_score(y, pred, average="macro")


optuna.logging.set_verbosity(optuna.logging.WARNING)
study = optuna.create_study(direction="maximize", sampler=TPESampler())
study.optimize(objective, n_trials=100)

study.best_params
