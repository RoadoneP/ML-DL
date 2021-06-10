import sys, os #sys는 하드웨어적인 메모리 등에 무언가의 영향을 행사하기 위해 주로 쓴다
# os모듈은 운영 체제와 상호 작용하기 위한 수십 가지 함수
sys.path.append(os.pardir) #os.pardir 현재 디렉토리의 부모 디렉토리를 가리킨다.
import numpy as np
from common.layers import * #Relu, Sigmoid, Affine, SoftmaxWithLoss, Dropout 등등
from common.gradient import numerical_gradient #기울기 구현 함수
from collections import OrderedDict #파이썬 딕셔너리와 비슷하지만, 정렬된 딕셔너리를 만들 수 있음 파이썬 딕셔너리는 순서를 관리하지X


class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        # 가중치 초기화 
        self.params = {}
        # 표준편차 값 * 행렬 표준편차 값이 0이되면 모든 가중치가 균일하게 갱신
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)
        
        #Build Layers
        self.layers = OrderedDict()
        self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
        self.layers['Relu1'] = Relu()
        self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])

        self.lastLayer = SoftmaxWithLoss()

    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
        return x
    
    def loss(self, x, t):
        y = self.predict(x)
        return self.lastLayer.forward(y, t)

    def accuracy(self, x, t):
        y = self.predict(x) 
        y = np.argmax(y, axis=1) #가장 큰값의 인덱스를 반환 0.95이런 것의 실직적인 값!!
        if t.ndim != 1 : #차원
            t = np.argmax(t, axis=1)
        
        accuracy = np.sum(y==t) / float(x.shape[0])
        return accuracy
    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t) # lambda 인자 : 표현식
        
        grads = {}
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])
        return grads
    
    def gradient(self, x, t):
        # foward
        self.loss(x, t)
        
        # backward
        dout = 1
        dout = self.lastLayer.backward(dout)
        
        layers = list(self.layers.values())
        layers.reverse() # back이기 때문에
        for layer in layers:
            dout = layer.backward(dout)
        
        grads = {}
        grads['W1'] = self.layers['Affine1'].dW # self.dW = np.dot(self.x.T, dout)
        grads['b1'] = self.layers['Affine1'].db
        grads['W2'] = self.layers['Affine2'].dW
        grads['b2'] = self.layers['Affine2'].db
        return grads
