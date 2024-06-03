import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


IMG_DIR = Path("/Users/yusong/Documents/venture/datasets/trichotrack/raw")
ANT_DIR = Path("/Users/yusong/Documents/venture/datasets/trichotrack/annotation")
TOKEN_DIR = Path("/Users/yusong/Documents/dinov2/dev/vitl14-ensemble")


start_id, end_id = 0, 10000 - 1


idx2img_ids = []
sw_labels, sr_labels = [], []
for img_id in range(start_id, end_id + 1):
    with ANT_DIR.joinpath(f"{img_id}.json").open("r") as f:
        ant = json.load(f)
        if not ant["discarded"]:
            if ant["scalp_wrinkle"] is None:
                # print(f"Image {img_id} is not discarded while without annotations")
                continue
            idx2img_ids.append(img_id)
            sw_labels.append(ant["scalp_wrinkle"]["scalp_wrinkle"])
            sr_labels.append(ant["scalp_redness"]["scalp_redness"])
sw_labels, sr_labels = np.array(sw_labels), np.array(sr_labels)
# print(len(idx2img_ids))
sw_counts = dict(zip(*np.unique(sw_labels, return_counts=True)))
sr_counts = dict(zip(*np.unique(sr_labels, return_counts=True)))
print(sw_counts)
print(sr_counts)
sw_weights = np.array([1 / sw_counts[l] for l in sw_labels])
sr_weights = np.array([1 / sr_counts[l] for l in sr_labels])

sss = StratifiedShuffleSplit(n_splits=1, test_size=1000, random_state=42)
sw_train_idxes, sw_test_idxes = list(sss.split(np.reshape(idx2img_ids, (-1, 1)), sw_labels))[0]
sr_train_idxes, sr_test_idxes = list(sss.split(np.reshape(idx2img_ids, (-1, 1)), sr_labels))[0]
# print(len(sw_train_idxes), len(sw_test_idxes), set(sw_train_idxes).intersection(sw_test_idxes))
# print(np.unique(sw_labels[sw_train_idxes], return_counts=True))
# print(np.unique(sw_labels[sw_test_idxes], return_counts=True))
# print(len(sr_train_idxes), len(sr_test_idxes), set(sr_train_idxes).intersection(sr_test_idxes))
# print(np.unique(sw_labels[sr_train_idxes], return_counts=True))
# print(np.unique(sw_labels[sr_test_idxes], return_counts=True))

tokens = np.load(TOKEN_DIR.joinpath(f"condensed_{start_id}_{end_id}.npz"))
cls_tokens = tokens["cls_tokens"][idx2img_ids]
pat_tokens = tokens["pat_tokens"][idx2img_ids]
cls_tokens /= np.linalg.norm(cls_tokens, axis=1, keepdims=True)
pat_tokens /= np.linalg.norm(pat_tokens, axis=1, keepdims=True)


tokens = {"CLS": cls_tokens, "PAT": pat_tokens}
labels = {"SW": sw_labels, "SR": sr_labels}
weights = {"SW": sw_weights, "SR": sr_weights}
train_idxes = {"SW": sw_train_idxes, "SR": sr_train_idxes}
test_idxes = {"SW": sw_test_idxes, "SR": sr_test_idxes}

svm = SVC(
    C=8000,
    kernel="rbf",
    gamma="auto",  # "scale" improves std acc at sacrifice of bal acc.
    shrinking=True,
    class_weight="balanced",
    random_state=42,
)

knn = KNeighborsClassifier(
    n_neighbors=10,
    weights="uniform",
    algorithm="brute",
    metric="cosine",
    n_jobs=-1,
)

classifiers = {
    "KNN": knn,
    "SVM": svm,
}
for clf_tag, clf in classifiers.items():
    for token_tag, x in tokens.items():
        for task_tag, y in labels.items():
            _train_idxes = train_idxes[task_tag]
            _test_idxes = test_idxes[task_tag]
            w = weights[task_tag]
            clf.fit(
                X=x[_train_idxes],
                y=y[_train_idxes],
            )
            std_acc = clf.score(
                X=x[_test_idxes],
                y=y[_test_idxes],
            )
            bal_acc = clf.score(
                X=x[_test_idxes],
                y=y[_test_idxes],
                sample_weight=w[_test_idxes],
            )
            print(f"{clf_tag} - {token_tag} - {task_tag} - Std - {std_acc:.2%} - Bal - {bal_acc:.2%}")
