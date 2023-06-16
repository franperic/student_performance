import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from sklearn.metrics import classification_report
from sklearn.model_selection import GroupKFold
import optuna
from optuna.samplers import TPESampler

from xgboost import XGBClassifier


df = pd.read_csv("data/train.csv")
labels = pd.read_csv("data/train_labels.csv")
# split column by underscore
# labels[["session_id", "question"]] = labels.session_id.str.split("_", expand=True)
# labels.question = labels.question.str.replace("q", "").astype(int)

labels = labels.assign(
    session=lambda x: x.session_id.str.split("_", expand=True)[0].astype(int),
    question=(
        lambda x: x.session_id.str.split("_", expand=True)[1]
        .str.replace("q", "")
        .astype(int)
    ),
)

# split data into train and validation sets
sessions = df.session_id.unique()
np.random.seed(42)
val_sessions = np.random.choice(sessions, size=int(len(sessions) * 0.2), replace=False)
train_sessions = np.setdiff1d(sessions, val_sessions)

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

train_X = df.loc[df.session_id.isin(train_sessions)]
train_y = labels.loc[labels.session.isin(train_sessions)]
val_X = df.loc[df.session_id.isin(val_sessions)]
val_y = labels.loc[labels.session.isin(val_sessions)]


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


train_X = feature_prep(train_X)
val_X = feature_prep(val_X)


gkf = GroupKFold(n_splits=5)
oof = pd.DataFrame(data=np.zeros((train_X.session_id.nunique(), 18)))
oof.index = train_X.session_id.unique()


models = {}
best_params = {}
for i, (train_index, test_index) in enumerate(
    gkf.split(X=train_X, groups=train_X.session_id)
):
    print("#" * 25)
    print("### Fold", i + 1)
    print("#" * 25)

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
        "use_label_encoder": False,
    }

    # ITERATE THRU QUESTIONS 1 THRU 18
    for q in range(1, 19):
        # USE THIS TRAIN DATA WITH THESE QUESTIONS
        if q <= 3:
            grp = "0-4"
        elif q <= 13:
            grp = "5-12"
        elif q <= 18:
            grp = "13-22"

        # TRAIN DATA
        train_X_q = train_X.iloc[train_index]
        train_X_q = train_X_q.loc[train_X_q.level_group == grp]
        train_users = train_X_q.session_id.values
        train_y_q = (
            train_y.loc[train_y.question == q].set_index("session").loc[train_users]
        )

        # VALID DATA
        valid_X = train_X.iloc[test_index]
        valid_X = valid_X.loc[valid_X.level_group == grp]
        valid_users = valid_X.session_id.values
        valid_y = (
            train_y.loc[train_y.question == q].set_index("session").loc[valid_users]
        )

        # TRAIN MODEL
        clf = XGBClassifier(**xgb_params)
        clf.fit(
            train_X_q.drop(["session_id", "level_group"], axis=1).astype("float32"),
            train_y_q["correct"],
            eval_set=[
                (
                    valid_X.drop(["session_id", "level_group"], axis=1).astype(
                        "float32"
                    ),
                    valid_y["correct"],
                )
            ],
            verbose=0,
        )
        print(f"{q}({clf.best_ntree_limit}), ", end="")

        # SAVE MODEL, PREDICT VALID OOF
        models[f"{grp}_{q}"] = clf
        oof.loc[valid_users, q - 1] = clf.predict_proba(
            valid_X.drop(["session_id", "level_group"], axis=1).astype("float32")
        )[:, 1]

    print()


true = oof.copy()
for k in range(18):
    tmp = (
        labels.loc[labels.question == k + 1]
        .set_index("session")
        .loc[train_X.session_id.unique()]
    )
    true[k] = tmp.correct.values

scores = []
cutoffs = []
best_score = 0
best_cutoff = 0

for cutoff in np.arange(0.3, 0.85, 0.01):
    print(f"{cutoff:.02f}, ", end="")
    preds = (oof.values.reshape((-1)) > cutoff).astype(int)
    m = f1_score(true.values.reshape((-1)), preds, average="macro")
    scores.append(m)
    cutoffs.append(cutoff)
    if m > best_score:
        best_score = m
        best_cutoff = cutoff


# PLOT THRESHOLD VS. F1_SCORE
plt.figure(figsize=(20, 5))
plt.plot(cutoffs, scores, "-o", color="blue")
plt.scatter([best_cutoff], [best_score], color="blue", s=300, alpha=1)
plt.xlabel("Threshold", size=14)
plt.ylabel("Validation F1 Score", size=14)
plt.title(
    f"Threshold vs. F1_Score with Best F1_Score = {best_score:.3f} at Best Threshold = {best_cutoff:.3}",
    size=18,
)
plt.show()


print("When using optimal threshold...")
for k in range(18):
    # COMPUTE F1 SCORE PER QUESTION
    m = f1_score(
        true[k].values, (oof[k].values > best_cutoff).astype("int"), average="macro"
    )
    print(f"Q{k}: F1 =", m)

# COMPUTE F1 SCORE OVERALL
m = f1_score(
    true.values.reshape((-1)),
    (oof.values.reshape((-1)) > best_cutoff).astype("int"),
    average="macro",
)
print("==> Overall F1 =", m)


val_X
val_y

for case in models.keys():
    grp, q = case.split("_")
    model = models[case]
    val_X_q = val_X.loc[val_X.level_group == grp]
    val_y_q = val_y.loc[val_y.question == int(q), "correct"]
    val_preds = model.predict_proba(
        val_X_q.drop(["session_id", "level_group"], axis=1).astype("float32")
    )[:, 1]


def find_cutoff(
    model: XGBClassifier, actuals: pd.Series, predictions: pd.Series
) -> float:
    y = actuals
    y_pred = predictions

    def objective(trial):
        cutoff = trial.suggest_float("cutoff", 0.1, 0.9)
        pred = y_pred > cutoff
        return f1_score(y, pred)

    study = optuna.create_study(direction="maximize", sampler=TPESampler())
    study.optimize(objective, n_trials=100)

    return study.best_params
