import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

bream_length = [25.4, 26.3, 26.5, 29.0, 29.0, 29.7, 29.7, 30.0, 30.0, 30.7, 31.0, 31.0, 31.5, 32.0, 32.0, 32.0,
                33.0, 33.0, 33.5, 33.5, 34.0, 34.0, 34.5, 35.0, 35.0, 35.0, 35.0, 36.0, 36.0, 37.0, 38.5, 38.5,
                39.5, 41.0, 41.0]
print(len(bream_length))
bream_weight = [242.0, 290.0, 340.0, 363.0, 430.0, 450.0, 500.0, 390.0, 450.0, 500.0, 475.0, 500.0, 500.0, 340.0,
                600.0, 600.0, 700.0, 700.0, 610.0, 650.0, 575.0, 685.0, 620.0, 680.0, 700.0, 725.0, 720.0, 714.0,
                850.0, 1000.0, 920.0, 955.0, 925.0, 975.0, 950.0]
print(len(bream_weight))

fish_data = np.column_stack((bream_length, bream_weight))
print(len(fish_data))
# print(fish_data)

# fish_target = np.concatenate((np.ones(35), np.zeros(14)))
fish_target = np.concatenate((np.ones(21), np.zeros(14)))
print(len(fish_target))
# print(fish_target)
train_input, test_input, train_target, test_target = train_test_split(fish_data, fish_target, random_state=42)
# print(train_input.shape)
# print(test_input.shape)
# print(train_target.shape)
# print(test_target.shape)
print(train_input[:5])
print(test_input[:5])
print(train_target[:5])
print(test_target[:5])

kn = KNeighborsClassifier()
kn.fit(fish_data, fish_target)
predictValue = kn.predict([[25, 150]])
print('예상값', predictValue)

# plt.scatter(train_input[:, 0], train_input[:, 1], c='r')
# plt.scatter(25, 150, c='b', marker='^')
plt.xlabel('length')
plt.ylabel('weight')

distance, indexes = kn.kneighbors([[25, 150]])
# print(distance)
# print(indexes)
# print('5개 요소의 x,y 좌표들', train_input[indexes, 0], train_input[indexes, 1])
# plt.scatter(fish_data[indexes, 0], fish_data[indexes, 1], marker='D')
# plt.xlim(0, 500)

mean = np.mean(train_input, axis=0)
std = np.std(train_input, axis=0)
print('평균', mean)
print('분산', std)

train_scale = train_input - (mean / std)

print(train_scale, len(train_scale))
plt.scatter(train_scale[:, 0], train_scale[:, 1])
plt.show()
