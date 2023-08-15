import tensorflow as tf

class Spline_tri():
    def __init__(self,x,x_target):
        
        self.h = tf.experimental.numpy.diff(x)
        self.n = x.shape[0]
        self.a = tf.stack([self.h[i]/(self.h[i]+self.h[i+1]) for i in range(self.n-2)] + [0], axis=0)
        self.b = tf.scalar_mul(2,tf.ones(x.shape[0]))
        self.c = tf.stack([0]+[self.h[i+1]/(self.h[i]+self.h[i+1]) for i in range(self.n-2)], axis=0)
        self.y_res = tf.Variable(initial_value=tf.zeros((x_target.shape[0],)),dtype=tf.float32)
        self.coef = tf.Variable(initial_value=tf.zeros((x.shape[0]-1,4)),dtype=tf.float32)
        self.diagonals = (self.c,self.b,self.a)
        self.x = x
        self.x_target = x_target

        self.interval=tf.Variable(initial_value=tf.zeros((self.n-1,2),dtype=tf.int32))
        for i in tf.range(self.n-1):
            if tf.equal(i,0):
                self.interval[i,:].assign([tf.where(x_target < x[i+1])[0,0],tf.where(x_target < x[i+1])[-1,0]+1])
            elif tf.equal(i,self.n-2):
                self.interval[i,:].assign([tf.where(x_target >= x[i])[0,0],tf.where(x_target >= x[i])[-1,0]+1])
            else:
                boolean_array = tf.logical_and(x_target >= x[i], x_target < x[i+1])
                self.interval[i,:].assign([tf.where(boolean_array)[0,0],tf.where(boolean_array)[-1,0]+1])

    @tf.function
    def do_spline(self, y):
        y_res = []
        yp = tf.pad(y[:, 1:  ], tf.constant([[0,0],[0,1]]), constant_values=1)
        ym = tf.pad(y[:,  :-1], tf.constant([[0,0],[1,0]]), constant_values=1)
        hp = tf.pad(self.h,  tf.constant([[0,1]]), constant_values=1)
        hm = tf.pad(self.h,  tf.constant([[1,0]]), constant_values=1)

        rhs = (yp-y)/hp - (y-ym)/hm
        rhs /= hp+hm
        rhs *= 6
        
        rhs = tf.pad(rhs[:, 1:-1],tf.constant([[0,0],[1,1]]))
        X = tf.transpose(tf.linalg.tridiagonal_solve(self.diagonals, tf.transpose(rhs), diagonals_format='sequence'))
        for i in range(self.n-1):
            C = tf.concat([[(X[:,i+1]-X[:,i])*self.h[i]*self.h[i]/6],
                            [X[:,i]*self.h[i]*self.h[i]/2],
                            [(y[:,i+1] - y[:,i] - (X[:,i+1]+2*X[:,i])*self.h[i]*self.h[i]/6)],
                            [y[:,i]]],0)
            z = tf.divide(tf.subtract(self.x_target[self.interval[i,0]:self.interval[i,1]],self.x[i]),self.h[i])
            Z = tf.concat([[tf.pow(z,3)],[tf.pow(z,2)],[tf.pow(z,1)],[tf.pow(z,0)]],0)
            y_res.append(tf.linalg.matmul(C,Z,transpose_a=True))
        return tf.concat(y_res,1)


class Spline():
    def __init__(self,x,x_target):
        self.N = x.shape[0] - 1
        A = tf.Variable(initial_value=tf.zeros((4*self.N,4*self.N)),dtype=tf.float32)
        self.x = x
        self.x_target = x_target

        for i in tf.range(self.N):
            A[2*i,     4*i+0].assign(tf.pow(x[i],3))
            A[2*i,     4*i+1].assign(tf.pow(x[i],2))
            A[2*i,     4*i+2].assign(x[i])
            A[2*i,     4*i+3].assign(1)

            A[2*i + 1, 4*i+0].assign(tf.pow(x[i+1],3))
            A[2*i + 1, 4*i+1].assign(tf.pow(x[i+1],2))
            A[2*i + 1, 4*i+2].assign(x[i+1])
            A[2*i + 1, 4*i+3].assign(1)

            if tf.not_equal(i,self.N-1):
                A[2*self.N + i,     4*i+0    ].assign(3*tf.pow(x[i+1],2))
                A[2*self.N + i,     4*i+1    ].assign(2*x[i+1])
                A[2*self.N + i,     4*i+2    ].assign(1)

                A[2*self.N + i,     4*(i+1)+0].assign(-3*tf.pow(x[i+1],2))
                A[2*self.N + i,     4*(i+1)+1].assign(-2*x[i+1])
                A[2*self.N + i,     4*(i+1)+2].assign(-1)

                A[3*self.N - 1 + i, 4*i+0    ].assign(6*x[i+1])
                A[3*self.N - 1 + i, 4*i+1    ].assign(2)

                A[3*self.N - 1 + i, 4*(i+1)+0].assign(-6*x[i+1])
                A[3*self.N - 1 + i, 4*(i+1)+1].assign(-2)

        A[4*self.N-2, 0        ].assign(6*x[0])
        A[4*self.N-2, 1        ].assign(2)
        A[4*self.N-1, 4*(self.N-1)  ].assign(6*x[self.N])
        A[4*self.N-1, 4*(self.N-1)+1].assign(2)

        self.lu, self.p = tf.linalg.lu(A)

        self.interval=tf.Variable(initial_value=tf.zeros((self.N,2),dtype=tf.int32))
        for i in tf.range(self.N):
            if tf.equal(i,0):
                self.interval[i,:].assign([tf.where(x_target < x[i+1])[0,0],tf.where(x_target < x[i+1])[-1,0]+1])
            elif tf.equal(i,self.N-1):
                self.interval[i,:].assign([tf.where(x_target >= x[i])[0,0],tf.where(x_target >= x[i])[-1,0]+1])
            else:
                boolean_array = tf.logical_and(x_target >= x[i], x_target < x[i+1])
                self.interval[i,:].assign([tf.where(boolean_array)[0,0],tf.where(boolean_array)[-1,0]+1])

    @tf.function
    def do_spline(self,y):
        y_res=[]
        P = []
        for i in range(self.N):
            P.append(y[:,i:i+1])
            P.append(y[:,i+1:i+2])
        Y = tf.concat(P,1)
        Y = tf.pad(Y,[[0,0],[0,self.N*2]])
        Y = tf.transpose(Y)
        X = tf.linalg.lu_solve(self.lu,self.p,Y)
        for i in range(self.N):
            z = self.x_target[self.interval[i,0]:self.interval[i,1]]
            Z = tf.concat([[tf.pow(z,3)],[tf.pow(z,2)],[tf.pow(z,1)],[tf.pow(z,0)]],0)
            y_res.append(tf.linalg.matmul(X[4*i+0:4*i+4,:],Z,transpose_a=True))
        return tf.concat(y_res,1)
