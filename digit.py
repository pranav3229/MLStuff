import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import seaborn as sn

digits = load_digits()
print(digits.data[0])

plt.gray()
for i in range(5):
    plt.matshow(digits.images[i], cmap='gray')
    # plt.show()
print(digits.target[0:5])    

X_train, x_test, y_train,y_test=train_test_split(digits.data,digits.target,test_size=0.2)
model =LogisticRegression()
model.fit(X_train,y_train)
y_predicted = model.predict(x_test)

print(model.score(x_test,y_test))
# plt.matshow.digits.images[67]
print(model.predict(digits.data[0:5]))

cm=confusion_matrix(y_test,y_predicted)
print(cm)

plt.figure(figsize=(10,7))
sn.heatmap(cm,annot=True)
plt.xlabel('Predicted')
plt.ylabel('Truth')

plt.show()
