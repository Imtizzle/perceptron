import cPickle as pk, matplotlib.pyplot as mp, numpy as np, random

#Using Perceptron learning rule to classify points

class Perceptron:

    def __init__(self,data,l):
        self.labels = data['labels']
        self.x = data['vectors'][0]
        self.y = data['vectors'][1]
        self.lr = l
        self.weightvec = np.random.rand(2)
        self.bias = 0.0
        self.right = []

    def test(self):
        self.wrong = []
        for i in xrange(len(self.labels)):
            j = self.x[i]*self.weightvec[0] + self.y[i]*self.weightvec[1] < self.bias
            if j != self.labels[i][0]:
                self.wrong.append((i,j))

    def test_iterations(self,n):
        self.iteration_size = n
        for iteration_number in xrange(n):
            self.test()
            self.right.append(len(self.labels) - len(self.wrong))
            i,j = random.choice(self.wrong)
            error = self.labels[i][0] - j
            self.weightvec[0] += self.lr*error*self.x[i]
            self.weightvec[1] += self.lr*error*self.y[i]

    def iteration_plot(self,n,save):
        self.test_iterations(n)
        vector = self.weightvec

        colours = ['b' if i == 1 else 'r' for i in self.labels]
        mp.plot((vector[1],-vector[1]),(-vector[0],vector[0]),label=r'$w^\bot$')
        mp.scatter(self.x,self.y,c=colours,label='data points')
        mp.xlim(-15,15)
        mp.ylim(-5,5)
        mp.xlabel('x')
        mp.ylabel('y')
        mp.title('Perceptron Classification with %d Iterations'%self.iteration_size)
        mp.legend()
        if save == True:
            mp.savefig('Perceptron Data Classification, %d iterations.png'%self.iteration_size)
        else: mp.show()