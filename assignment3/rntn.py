import numpy as np
import collections
np.seterr(over='raise',under='raise')

class RNTN:

    def __init__(self,wvecDim,outputDim,numWords,mbSize=30,rho=1e-6):
        self.wvecDim = wvecDim
        self.outputDim = outputDim
        self.numWords = numWords
        self.mbSize = mbSize
        self.defaultVec = lambda : np.zeros((wvecDim,))
        self.rho = rho

    def initParams(self):
        np.random.seed(12341)
        
        # Word vectors
        self.L = 0.01*np.random.randn(self.wvecDim,self.numWords)

        # Hidden activation weights
        self.V = 0.01*np.random.randn(self.wvecDim,2*self.wvecDim,2*self.wvecDim)
        self.W = 0.01*np.random.randn(self.wvecDim,self.wvecDim*2)
        self.b = np.zeros((self.wvecDim))

        # Softmax weights
        self.Ws = 0.01*np.random.randn(self.outputDim,self.wvecDim)
        self.bs = np.zeros((self.outputDim))

        self.stack = [self.L, self.V, self.W, self.b, self.Ws, self.bs]

        # Gradients
        self.dV = np.empty((self.wvecDim,2*self.wvecDim,2*self.wvecDim))
        self.dW = np.empty(self.W.shape)
        self.db = np.empty((self.wvecDim))
        self.dWs = np.empty(self.Ws.shape)
        self.dbs = np.empty((self.outputDim))

    def costAndGrad(self,mbdata,test=False): 
        """
        Computes cost and gradients for mini-batch data.
        Data is propagated and back-propagated in each tree.
        :param mini_batch_data: List of data pieces (i.e. trees).
        :return: Cost, Gradients of W, W_s, b, b_s, L.
        """

        cost, total = 0.0, 0.0
        correct = []
        guess = []
        self.L, self.V, self.W, self.b, self.Ws, self.bs = self.stack

        # Set gradients to zero.

        self.dV[:] = 0
        self.dW[:] = 0
        self.db[:] = 0
        self.dWs[:] = 0
        self.dbs[:] = 0

        self.dL = collections.defaultdict(self.defaultVec)

        # Propagate data in each tree in a mini-batch fashion.
        # --------------------------

        for tree in mbdata:
            c, tot = self.forwardProp(tree.root, correct, guess)
            cost += c
            total += tot
        if test:
            return (1./len(mbdata))*cost, correct, guess, total

        # Back-propagate data in each tree.
        # --------------------------
        for tree in mbdata:
            self.backProp(tree.root)

        # Scale cost and gradients by mini-bach size.
        # --------------------------
        scale = (1./self.mbSize)
        for v in self.dL.itervalues():
            v *= scale

        # Add L2 Regularization term.
        # --------------------------
        cost += self.rho / 2 * np.sum(self.V ** 2)
        cost += self.rho / 2 * np.sum(self.W ** 2)
        cost += self.rho / 2 * np.sum(self.Ws ** 2)

        return scale*cost, [self.dL, scale*(self.dV + self.rho * self.V),
                            scale * (self.dW + self.rho * self.W), scale * self.db,
                            scale * (self.dWs + self.rho * self.Ws), scale * self.dbs]

    def forwardProp(self,node, correct, guess):
        """
        Forward propagation at node.
        :return: (Cross-entropy cost,
        Number of correctly classified items,
        Number of classified items).
        """

        cost, total = 0.0, 0.0

        if node.isLeaf:
            # Hidden activations at leaves are occurences of self.word.
            node.hActs1 = self.L[:, node.word]
            node.fprop = True

        else:
            if not node.left.fprop:
                c, tot = self.forwardProp(node.left, correct, guess)
                cost += c
                total += tot

            if not node.right.fprop:
                c, tot = self.forwardProp(node.right, correct, guess)
                cost += c
                total += tot

            # Stack left-right children word vectors. Compute matrix operations for parent vector.
            lr = np.hstack([node.left.hActs1, node.right.hActs1])
            node.hActs1 = np.dot(self.W, lr) + self.b
            node.hActs1 += np.tensordot(self.V, np.outer(lr, lr), axes=([1, 2], [0, 1]))

            # Compute parent vector.
            node.hActs1 = np.tanh(node.hActs1)

        # Compute classification labels via softmax.
        node.probs = np.dot(self.Ws, node.hActs1) + self.bs
        node.probs -= np.max(node.probs)
        node.probs = np.exp(node.probs)
        node.probs = node.probs/np.sum(node.probs)
        node.fprop = True

        correct.append(node.label)
        guess.append(np.argmax(node.probs))

        return cost - np.log(node.probs[node.label]), total + 1


    def backProp(self,node,error=None):

        """
        Backward propagation in node.
        """

        # Clear node.
        node.fprop = False

        # Softmax gradients.
        # --------------------------
        softmax_node_error = node.probs  # Predicted distribution
        softmax_node_error[node.label] -= 1.0  # Targeted distribution equals 1 for node.label, else 0.

        self.dWs += np.outer(softmax_node_error, node.hActs1)
        self.dbs += softmax_node_error
        softmax_node_error = np.dot(self.Ws.T, softmax_node_error)

        if error is not None:
            # To back-propagate error recursively
            softmax_node_error += error

        softmax_node_error *= (1 - node.hActs1 ** 2)

        # Update L at leaf nodes.
        if node.isLeaf:
            self.dL[node.word] += softmax_node_error
            return

        # Hidden gradients.
        # --------------------------
        if not node.isLeaf:
            lr = np.hstack([node.left.hActs1, node.right.hActs1])  # Left-right stacked activation
            outer = np.outer(softmax_node_error, lr)

            self.dV += (np.outer(lr, lr)[:, :, None] * softmax_node_error).T
            self.dW += outer
            self.db += softmax_node_error

            # Compute error for children.
            softmax_node_error = np.dot(self.W.T, softmax_node_error)
            softmax_node_error += np.tensordot(self.V.transpose((0, 2, 1)) + self.V,
                                               outer.T, axes=([1, 0], [0, 1]))
            self.backProp(node.left, softmax_node_error[:self.wvecDim])
            self.backProp(node.right, softmax_node_error[self.wvecDim:])

        
    def updateParams(self,scale,update,log=False):
        """
        Updates parameters as
        p := p - scale * update.
        If log is true, prints root mean square of parameter
        and update.
        """
        if log:
            for P,dP in zip(self.stack[1:],update[1:]):
                pRMS = np.sqrt(np.mean(P**2))
                dpRMS = np.sqrt(np.mean((scale*dP)**2))
                print "weight rms=%f -- update rms=%f"%(pRMS,dpRMS)

        self.stack[1:] = [P+scale*dP for P,dP in zip(self.stack[1:],update[1:])]

        # handle dictionary update sparsely
        dL = update[0]
        for j in dL.iterkeys():
            self.L[:,j] += scale*dL[j]

    def toFile(self,fid):
        import cPickle as pickle
        pickle.dump(self.stack,fid)

    def fromFile(self,fid):
        import cPickle as pickle
        self.stack = pickle.load(fid)

    def check_grad(self,data,epsilon=1e-6):

        cost, grad = self.costAndGrad(data)
        err1 = 0.0
        count = 0.0

        print "Checking dW... (might take a while)"
        for W,dW in zip(self.stack[1:],grad[1:]):
            W = W[...,None,None] # add dimension since bias is flat
            dW = dW[...,None,None] 
            for i in xrange(W.shape[0]):
                for j in xrange(W.shape[1]):
                    for k in xrange(W.shape[2]):
                        W[i,j,k] += epsilon
                        costP,_ = self.costAndGrad(data)
                        W[i,j,k] -= epsilon
                        numGrad = (costP - cost)/epsilon
                        err = np.abs(dW[i,j,k] - numGrad)
                        #print "Analytic %.9f, Numerical %.9f, Relative Error %.9f"%(dW[i,j,k],numGrad,err)
                        err1+=err
                        count+=1
        if 0.001 > err1/count:
            print "Grad Check Passed for dW"
        else:
            print "Grad Check Failed for dW: Sum of Error = %.9f" % (err1/count)

        # check dL separately since dict
        dL = grad[0]
        L = self.stack[0]
        err2 = 0.0
        count = 0.0
        print "Checking dL..."
        for j in dL.iterkeys():
            for i in xrange(L.shape[0]):
                L[i,j] += epsilon
                costP,_ = self.costAndGrad(data)
                L[i,j] -= epsilon
                numGrad = (costP - cost)/epsilon
                err = np.abs(dL[j][i] - numGrad)
                #print "Analytic %.9f, Numerical %.9f, Relative Error %.9f"%(dL[j][i],numGrad,err)
                err2+=err
                count+=1

        if 0.001 > err2/count:
            print "Grad Check Passed for dL"
        else:
            print "Grad Check Failed for dL: Sum of Error = %.9f" % (err2/count)

if __name__ == '__main__':

    import tree as treeM
    train = treeM.loadTrees()
    numW = len(treeM.loadWordMap())

    wvecDim = 10
    outputDim = 5

    nn = RNTN(wvecDim,outputDim,numW,mbSize=4)
    nn.initParams()

    mbData = train[:1]
    #cost, grad = nn.costAndGrad(mbData)

    print "Numerical gradient check..."
    nn.check_grad(mbData)






