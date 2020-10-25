import numpy as np
import scipy.signal as sc
import scipy.stats as st
import matplotlib.pyplot as plt
import pandas as pd

# # numer1
# plt.figure(1, figsize=(15, 10))
# plt.subplot(2, 2, 1).set_title('Chirp')
#
# t = np.linspace(0, 10, 10000)
# w = sc.chirp(t, f0=4, f1=1, t1=10, method='linear')
# plt.plot(t, w)
#
# plt.subplot(2, 2, 2).set_title('Gausspulse')
# t = np.linspace(-1, 1, 2*100, endpoint=False)
# i, e, q = sc.gausspulse(t, fc=4, retquad=True, retenv=True)
# plt.plot(t, i, t, q, '--', t, e)
#
# plt.subplot(2, 2, 3).set_title('Square')
# t = np.linspace(0, 1, 1000, endpoint=False)
# w = sc.square(2*np.pi*4*t)
# plt.plot(t, w)
#
# plt.subplot(2, 2, 4).set_title('Sawtooth')
# t = np.linspace(0, 1, 1000, endpoint=False)
# w = sc.sawtooth(2*np.pi*5*t)
# plt.plot(t, w)
# plt.show()
#
# # numer2
# #brown (red) noise
# plt.figure(2, figsize=(15, 10))
#
# def red_noise(x0, n, dt, delta, out=None):
#     x0 = np.asarray(x0)
#     r = st.norm.rvs(size=x0.shape + (n,), scale=delta*np.sqrt(dt))
#     if out is None:
#         out = np.empty(r.shape)
#
#     np.cumsum(r, axis=-1, out=out)
#     out += np.expand_dims(x0, axis=-1)
#     return out
#
#
# delta = 4
# T = 10.0
# N = 500
# dt = T/N
# m = 1
# x = np.empty((m, N+1))
# x[:, 0] = 50
#
# red_noise(x[:, 0], N, dt, delta, out=x[:, 1:])
# plt.subplot(2, 1, 1).set_title('Brown noise periodogram')
# f, Pxx_den = sc.periodogram(x[0], 1/T)
# plt.plot(f, Pxx_den)
#
# #pink noise
# import colorednoise as cn
# beta = 1
# samples = 2 ** 10
# y = cn.powerlaw_psd_gaussian(beta, samples)
#
# plt.subplot(2, 1, 2).set_title('Pink noise periodogram')
# f, Pxx_den = sc.periodogram(y)
# plt.plot(f, Pxx_den)
# plt.show()
#
# # numer 3
# def sinusx3(x):
#     return np.sin(np.sin(x) -np.pi/2) * np.sin(x/2-np.pi/4)
# x = np.linspace(0, 10*np.pi, 1000)
# y = sinusx3(x)
# peaks, _ = sc.find_peaks(y, distance=50)
# print(np.amax(y[peaks]))
#
# plt.figure(3, figsize=(10, 5))
# plt.plot(x, y)
# plt.plot(x[peaks], y[peaks], 'o')
# plt.show()
from itertools import chain, zip_longest

data = pd.read_csv("data.csv")
data1 = data.drop(data.columns[[0,1]],axis=1)
data1.to_csv("data1.csv")
# data['HTR'] = ['D','W','L','W','W','W','L','W','W','W','W','L','L','D','W']
# data['ATR'] = ['D','L','W','L','L','L','W','L','L','L','L','W','W','D','L']
# X = data[['HS', 'AS']].values
# Y = data[['HTHG', 'FTAG']].values
# ftrH = data['HTR'].to_list()
# ftrA = data['ATR'].to_list()
# ftr = list(chain.from_iterable(zip_longest(ftrH, ftrA)))
# resKolor = ['tab:red' if p=='L' else 'tab:green' if p=='W' else 'tab:blue' for p in ftr]
# plt.scatter(X, Y, marker='o', facecolors=resKolor)
# plt.xlabel('Strikes', fontsize=16)
# plt.ylabel('Gols', fontsize=16)
# plt.show()

# Regression

# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression
# from sklearn.svm import SVR
#
# X = X.reshape(-1,1)
# Y = Y.reshape(-1,1)
# X_tren, X_test, Y_tren, Y_test = train_test_split(X, Y)
# linreg = LinearRegression().fit(X_tren,Y_tren)
# plt.scatter(X, Y, marker='o', facecolors='white', edgecolors='black', label='pomiary')
# plt.xlabel('Strikes', fontsize=16)
# plt.ylabel('Gols', fontsize=16)
# plt.scatter(X,linreg.predict(X), facecolor='tab:red', label='regresja liniowa')
# svmreg = SVR(gamma='auto').fit(X_tren,Y_tren.ravel())
# plt.scatter(X, svmreg.predict(X), facecolor='tab:green', label='regresja SVR')
# plt.legend()
# plt.show()
# from sklearn.metrics import mean_squared_error
# print('blad sredniokwadratowy na zbiorze treningowym: ', mean_squared_error(Y_tren, svmreg.predict(X_tren)))
# print('blad sredniokwadratowy na zbiorze testujacym: ', mean_squared_error(Y_test, svmreg.predict(X_test)))

# Klassification

# XX = np.array(data[['HS', 'AS']])
# XX = XX.reshape(-1,2)
# ftResult = data['FTR'].to_list()
# frKolor = ['tab:red' if p=='H' else 'tab:green' if p=='A' else 'tab:blue' for p in ftResult]
# XX_tren, XX_test, Y_tren, Y_test = train_test_split(XX, frKolor)
# from sklearn.neighbors import KNeighborsClassifier
# knn = KNeighborsClassifier(n_neighbors=1)
# knn.fit(XX_tren, Y_tren)
# # XX_nowy = np.array([[15, 10]])
# # pred = knn.predict(XX_nowy)
# # print(pred)
# Y_pred = knn.predict(XX_test)
# print(Y_pred)
# print(np.mean(Y_pred == Y_test))
# print(knn.score(XX_test, Y_test))