import numpy as np

class Perceptron(object):
    """Implements a perceptron network"""
    def __init__(self, input_size, lr = 1, epochs = 6):
        self.W = np.zeros(input_size + 1)
        self.epochs = epochs
        self.lr = lr
    def activation_fn(self, x):
        return 1 if x >= 0 else 0
    def predict(self, x):
        x = np.insert(x, 0, 1)
        z = self.W.T.dot(x)
        a = self.activation_fn(z)
        return a
    def fit(self, X, d):
        for _ in range(self.epochs): 
            for i in range(d.shape[0]):
                y = self.predict(X[i])
                e = d[i] - y
                self.W = self.W + self.lr * e * np.insert(X[i], 0, 1)
                
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1] 
    ])
d = np.array([0,0,0,1])

perceptron=Perceptron(input_size=2)
perceptron.W

perceptron.fit(X,d)
perceptron.W

print(perceptron.W)

mp=Perceptron(5)
x=np.asarray([-10,-2,-30,4,-50])
mp.predict(x)

mp.activation_fn(-10)

# 1- activation_fn :ağırlıkları hesaplanarak fonksiyona gönderilen girdiler 0'dan büyük veya eşitse 1, değilse 0 döndürür. 
#    predict :perceptron aracılığıyla bir girdi vererek çıktı döndürebilmek için ihtiyacımız olan fonksiyondur. Bias değerini X girdilerine ekler. Ağırlıklar ile çarpım işlemlerini yaparak üretilen z değerini aktivasyon fonksiyonuna yollar.
#    fit : parametre olarak aldığı X ve d değerlerini kullanarak her bir döngüde e(error) değerinide hesaplayarak yeni ağırlıklar elde eder. e değeri 0'a yaklaştıkça doğru tahmin oranı artar.


# 2- XOR işlemi için 
#   d=np.array([0,1,1,0]) 
#   d değerleri bu şekilde güncellenir.
#   fonksiyonlar çalıştırıldıktan sonra yeni W değeri array([ 0., -1.,  0.]) bu şekilde olur.
#   ancak predict fonksiyonu üzerinden çıktı aldığımızda doğru değerleri üretmemektedir.
