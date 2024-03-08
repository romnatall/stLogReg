
import numpy as np

class LogReg:
    def __init__(self, learning_rate, n_inputs):
        self.learning_rate = learning_rate
        self.n_inputs = n_inputs
        self.coef_ = np.random.rand(n_inputs)
        self.intercept_ = np.random.rand()
        
    def fit(self, X, y,epochs=100):

        assert X.shape[0] == y.shape[0]
        assert X.shape[1] == self.n_inputs
        assert y.shape[1] == 1

        exp =np.exp
        w=self.coef_
        w0=self.intercept_
        for i in range(epochs):
            # gradw=-X*y* X*(1 - y)*exp(w*X + w0)/(1 - exp(w*X + w0))
            # gradi=-y + (1 - y)*exp(w*X + w0)/(1 - exp(w*X + w0))
            
            gradw=-X*y*exp(-X*w - w0)/(exp(-X*w - w0) + 1) + X*(1 - y)*exp(-X*w - w0)/((1 - 1/(exp(-X*w - w0) + 1))*(exp(-X*w - w0) + 1)**2)
            gradi=-y*exp(-X*w - w0)/(exp(-X*w - w0) + 1) + (1 - y)*exp(-X*w - w0)/((1 - 1/(exp(-X*w - w0) + 1))*(exp(-X*w - w0) + 1)**2)

            w -= self.learning_rate*np.mean(gradw, axis=0)
            w0 -= self.learning_rate* np.mean(gradi)

        self.coef_ = w
        self.intercept_ = w0
        
    def predict(self, X):

        return  1/(1+np.exp(- ( X@self.coef_  + self.intercept_)))
    
    def score(self, X, y):
        pred = self.predict(X)
        return np.mean(-( y* np.log(pred) + (1-y)*np.log(1-pred)))