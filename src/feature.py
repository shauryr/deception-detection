import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

path_file = 'profiles.csv'

df = pd.read_csv(open(path_file, 'r'))

list_df = []
df = df[df.status != "unknown"]
# df.dropna()
# df = df.rename(columns = {'fit': 'fit_feature'})

df['status'] = df['status'].replace("seeing someone", "married")
df['status'] = df['status'].replace("available", "single")

df['status'] = df['status'].replace("single", 0)
df['status'] = df['status'].replace("married", 1)

# df_single = df[df['status'] == 0].sample(frac=0.04124)
# df_married = df[df['status'] == 1]

# df = pd.concat([df_single, df_married], axis=1)

for columns in df.columns:
    if columns.startswith('essay') or columns.startswith('last_online') or columns.startswith('speaks') or columns.startswith('status'):
        continue
    else:
        list_df.append(pd.get_dummies(df[columns], prefix=columns))
        print(columns, pd.get_dummies(df[columns]).shape)

df_np = np.asarray(list_df)

features = pd.concat(df_np, axis=1)

labels = df['status']
feature_list = list(features.columns)

train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=0.25, random_state=42, stratify=labels)

print('Training Features Shape:', train_features.shape)
print('Training Labels Shape:', train_labels.shape)
print('Testing Features Shape:', test_features.shape)
print('Testing Labels Shape:', test_labels.shape)

rf = RandomForestClassifier(n_estimators=1000, random_state=42)

rf.fit(train_features, train_labels)
#
# for f in range(X.shape[1]):
#     print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

importances = rf.feature_importances_
std = np.std([tree.feature_importances_ for tree in rf.estimators_], axis=0)

indices = np.argsort(importances)[::-1]
print("Feature ranking:")

for f in range(train_features.shape[1]):
    print(f + 1, feature_list[indices[f]], importances[indices[f]])

plt.figure()
plt.title("Feature importances")
plt.bar(range(train_features.shape[1]), importances[indices],
        color="r", yerr=std[indices], align="center")
plt.xticks(range(train_features.shape[1]), indices)
plt.xlim([-1, train_features.shape[1]])
plt.show()

plt.figure()
plt.title("Feature importances")
plt.bar(range(train_features.shape[1])[:10], importances[indices][:10],
        color="r", yerr=std[indices][:10], align="center")
plt.xticks(range(train_features.shape[1])[:10], sort_features)
# plt.xlim([-1, train_features.shape[1]])
plt.show()

y_pred = rf.predict(test_features)
from sklearn.metrics import classification_report
target_names = ['single','married']
print(classification_report(test_labels, y_pred, target_names=target_names))