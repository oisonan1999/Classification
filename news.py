# Import thư viện
import copy
import numpy as np
import re
import nltk
nltk.download('stopwords')
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import VarianceThreshold
from imblearn.over_sampling import SMOTE
from sklearn.dummy import DummyClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
#from sklearn.metrics import accuracy_score
#from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


# # Chuẩn bị dữ liệu
# ## Tải dữ liệu
titles = []
categories = []
print("\nLoading data")
with open('dsjVoxArticles.tsv','r', encoding='utf-8') as tsv:
    count = 0;
    for line in tsv:
        a = line.strip().split('\t')[:3]
        if a[2] in ['Business & Finance', 'Health Care', 'Science & Health', 'Politics & Policy', 'Criminal Justice']:
            title = a[0].lower()
            title = re.sub('\s\W',' ',title)
            title = re.sub('\W\s',' ',title)
            titles.append(title)
            categories.append(a[2])

# ## Tách dữ liệu
print("\nSplitting data")
title_tr, title_te, category_tr, category_te = train_test_split(titles,categories)
title_tr, title_de, category_tr, category_de = train_test_split(title_tr,category_tr)
print("Training: ",len(title_tr))
print("Developement: ",len(title_de),)
print("Testing: ",len(title_te))




# # Xử lý dữ liệu 
# ## Vecto hóa dữ liệu
# Vecto hóa dữ liệu dùng Bag of words (BOW)
print("\nVectorizing data")
tokenizer = nltk.tokenize.RegexpTokenizer(r"\w+")
stop_words = nltk.corpus.stopwords.words("english")
vectorizer = CountVectorizer(tokenizer=tokenizer.tokenize, stop_words=stop_words)

vectorizer.fit(iter(title_tr))
Xtr = vectorizer.transform(iter(title_tr))
Xde = vectorizer.transform(iter(title_de))
Xte = vectorizer.transform(iter(title_te))

encoder = LabelEncoder()
encoder.fit(category_tr)
Ytr = encoder.transform(category_tr)
Yde = encoder.transform(category_de)
Yte = encoder.transform(category_te)

# ## Giảm chiều dữ liệu 
# Kiểm tra phương sai của tính năng và thả chúng dựa trên ngưỡng
print("\nApplyting Feature Reduction")
print("Number of features before reduction : ", Xtr.shape[1])
selection = VarianceThreshold(threshold=0.001)
Xtr_whole = copy.deepcopy(Xtr)
Ytr_whole = copy.deepcopy(Ytr)
selection.fit(Xtr)
Xtr = selection.transform(Xtr)
Xde = selection.transform(Xde)
Xte = selection.transform(Xte)
print("Number of features after reduction : ", Xtr.shape[1])

# ## Dữ liệu kiểm thử
sm = SMOTE(random_state=42)
Xtr, Ytr = sm.fit_sample(Xtr, Ytr)




# # Đào tạo mô hình    
# ### Mô hình cơ sở
#phân tầng: tạo dự đoán bằng cách tôn trọng phân phối lớp tập huấn.
print("\n\nTraining baseline classifier")
dc = DummyClassifier(strategy="stratified")
dc.fit(Xtr, Ytr)
pred = dc.predict(Xde)
print(classification_report(Yde, pred, target_names=encoder.classes_))

# ### Cây quyết định
print("Training Decision tree")
dt = DecisionTreeClassifier()
dt.fit(Xtr, Ytr)
pred = dt.predict(Xde)
print(classification_report(Yde, pred, target_names=encoder.classes_))

# ### Random Forest
print("Training Random Forest")
rf = RandomForestClassifier(n_estimators=40)
rf.fit(Xtr, Ytr)
pred = rf.predict(Xde)
print(classification_report(Yde, pred, target_names=encoder.classes_))

# ### Multinomial Naive Bayesian
print("Training Multinomial Naive Bayesian")
nb = MultinomialNB()
nb.fit(Xtr, Ytr)
pred_nb = nb.predict(Xde)
print(classification_report(Yde, pred_nb, target_names=encoder.classes_))

# ### Support Vector Classification
print("Training Support Vector Classification")
from sklearn.svm import SVC
svc = SVC()
svc.fit(Xtr, Ytr)
pred = svc.predict(Xde)
print(classification_report(Yde, pred, target_names=encoder.classes_))

# ### Multilayered Perceptron
print("Training Multilayered Perceptron")
mlp = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(100, 20), random_state=1, max_iter=400)
mlp.fit(Xtr, Ytr)
pred = mlp.predict(Xde)
print(classification_report(Yde, pred, target_names=encoder.classes_))



print("\n\nPredicting test data using Multinomial Naive Bayesian")
pred_final = nb.predict(Xte)
print(classification_report(Yte, pred_final, target_names=encoder.classes_))


# lấy dữ liệu phân loại không chính xác
print("\n\nIncorrectly classified")
incorrect = np.not_equal(pred_nb, Yde).nonzero()[0]
print(
    "\nTitle: ",titles[incorrect[1]],
    "\nTrue Category: ",categories[incorrect[1]],
    "\nPredicted Category: ", encoder.inverse_transform([pred[incorrect[1]]])[0]
)
