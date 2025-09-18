from sklearn.linear_model import LogisticRegression 
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier

SEED = 42

def make_logistic():
    return LogisticRegression(max_iter=1000, class_weight="balanced", random_state=SEED)

def make_tree(criterion="gini", max_depth=None, min_samples_leaf=1):
    return DecisionTreeClassifier(
        criterion = criterion,
        max_depth = max_depth,
        min_samples_leaf = min_samples_leaf,
        class_weight="balanced",
        random_state=SEED
    )

def make_bagging(n_estimators=100, base_max_depth=3):
    base = DecisionTreeClassifier(
        max_depth = base_max_depth,
        class_weight="balanced", 
        random_state=SEED
        )
    return BaggingClassifier(
        estimator=base,
        n_estimators=n_estimators,
        random_state=SEED,
        n_jobs=-1
    )

def make_rf(n_estimators=300, max_depth=None, min_samples_leaf=1, max_features="sqrt"):
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        class_weight="balanced",
        random_state=SEED,
        n_jobs=-1
    )
def make_adaboost(n_estimators=50, learning_rate=0.05):
    stump = DecisionTreeClassifier(
        max_depth=1,
        class_weight="balanced",
        random_state=SEED
    )
    return AdaBoostClassifier(
        estimator=stump,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        random_state=SEED
    )

def make_gb(n_estimators=250, learning_rate=0.05, max_depth=3, subsample=0.8):
    return GradientBoostingClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        subsample=subsample,
        random_state=SEED
    )
def make_voting():
    return VotingClassifier(
        estimators=[
            ("logistic", make_logistic()),
            ("rf", make_rf()),
            ("gb", make_gb())
        ],
        voting="soft",
        n_jobs=-1
    )

MODEL_BUILDERS = {
    "logistic": make_logistic,
    "tree": make_tree,
    "bagging": make_bagging,
    "rf": make_rf,
    "adaboost": make_adaboost,
    "gb": make_gb,
    "voting": make_voting
}