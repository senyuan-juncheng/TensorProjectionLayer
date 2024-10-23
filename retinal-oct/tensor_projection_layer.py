import tensorflow as tf

# Define TensorProjection Layer
class TensorProjectionLayer(tf.keras.layers.Layer):
    def __init__(self, q1, q2, q3, regularization='None', rate=0.1, **kwargs):
        super(TensorProjectionLayer, self).__init__(**kwargs)
        self.q1 = q1
        self.q2 = q2
        self.q3 = q3
        self.regularization = regularization
        self.rate = rate  # regularization coefficient
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.q1, self.q2, self.q3)
    
    def get_config(self):
        base_config = super(TensorProjectionLayer, self).get_config()
        base_config['q1'] = self.q1
        base_config['q2'] = self.q2
        base_config['q3'] = self.q3
        base_config['regularization'] = self.regularization
        base_config['rate'] = self.rate
        return base_config
    
    def build(self, input_shape):
        self.p1 = int(input_shape[1])
        self.p2 = int(input_shape[2])
        self.p3 = int(input_shape[3])
        
        if self.q1 < self.p1:
            self.W1 = self.add_weight(name="W1", shape=(self.p1, self.q1), initializer='normal', trainable=True)
        if self.q2 < self.p2:
            self.W2 = self.add_weight(name="W2", shape=(self.p2, self.q2), initializer='normal', trainable=True)
        if self.q3 < self.p3:
            self.W3 = self.add_weight(name="W3", shape=(self.p3, self.q3), initializer='normal', trainable=True)
            
        super(TensorProjectionLayer, self).build(input_shape)

    def kmode_product(self, T, A, k):
        n  = tf.shape(T)[0]
        A = tf.expand_dims(A, 0)
        An =  tf.tile(A, [n, 1, 1])
        if k == 1:
            return tf.einsum('npqr, nsp -> nsqr', T, An)
        elif k == 2:
            return tf.einsum('npqr, nsq -> npsr', T, An)
        elif k == 3:
            return tf.einsum('npqr, nsr -> npqs', T, An)
        
    def WtoU(self, W):
        e = 10**-6
        q = tf.shape(W)[1]
        Iq = tf.eye(q)
        WT = tf.transpose(W, perm=[1,0])
        M = tf.math.add(tf.linalg.matmul(WT, W), Iq * e)
        sqrtM = tf.linalg.sqrtm(M)
        G = tf.linalg.inv(sqrtM)
        U = tf.linalg.matmul(W, G)
        return U
    
    def call(self, X):
        Z = X
        n = tf.shape(X)[0]
        if self.q1 < self.p1:
            U1 = self.WtoU(self.W1)
            U1T = tf.transpose(U1, perm=[1,0])  # q1 x p1
            Z = self.kmode_product(Z, U1T, 1)
        if self.q2 < self.p2:
            U2 = self.WtoU(self.W2)
            U2T = tf.transpose(U2, perm=[1,0])  # q2 x p2
            Z = self.kmode_product(Z, U2T, 2)
        if self.q3 < self.p3:
            U3 = self.WtoU(self.W3)
            U3T = tf.transpose(U3, perm=[1,0])  # q3 x p3
            Z = self.kmode_product(Z, U3T, 3)
        
        if self.regularization == 'reconstruction_error':
            X_ = Z
            if self.q1 < self.p1:
                X_ = self.kmode_product(X_, U1, 1)
            if self.q2 < self.p2:
                X_ = self.kmode_product(X_, U2, 2)
            if self.q3 < self.p3:
                X_ = self.kmode_product(X_, U3, 3)
            dn2 = tf.math.squared_difference(X , X_)  # n, p1,p2,p3
            dn2 = tf.math.reduce_mean(dn2, axis=1)  # n, p2,p3
            dn2 = tf.math.reduce_mean(dn2, axis=1)  # n, p3
            dn2 = tf.math.reduce_mean(dn2, axis=1)  # n
            dn = tf.math.pow(dn2, 0.5)
            self.add_loss(self.rate * tf.math.log(tf.math.reduce_mean(dn)))
        elif self.regularization == 'total_variation':
            mz = tf.reduce_mean(Z, axis=0, keepdims=True)
            mz = tf.tile(mz, [n, 1, 1, 1])
            Z_ = Z - mz  # centerize
            v = tf.math.pow(Z_, 2)
            v = tf.reduce_mean(v, axis=1)
            v = tf.reduce_mean(v, axis=1)
            v = tf.reduce_mean(v, axis=1)
            v = tf.math.pow(v, 0.5)
            self.add_loss(self.rate * tf.math.log(tf.math.reduce_mean(v)))
            
        return Z
