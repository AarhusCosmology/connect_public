import tensorflow as tf

class Spline():
    def __init__(self, x, x_target):
        self.h = tf.experimental.numpy.diff(x)
        self.n = x.shape[0]
        self.a = tf.stack([self.h[i] / (self.h[i] + self.h[i + 1]) for i in range(self.n - 2)] + [0], axis=0)
        self.b = tf.scalar_mul(2, tf.ones(x.shape[0]))
        self.c = tf.stack([0] + [self.h[i + 1] / (self.h[i] + self.h[i + 1]) for i in range(self.n - 2)], axis=0)
        self.diagonals = (self.c, self.b, self.a)
        self.x = x
        self.x_target = x_target

        # Initialize interval as an empty tensor with the correct shape
        self.interval = tf.TensorArray(tf.int32, size=self.n - 1)

        # Define the loop body
        def loop_body(i, interval):
            def true_fn_0(interval):
                start = tf.where(x_target < x[i + 1])[0, 0]
                end = tf.where(x_target < x[i + 1])[-1, 0] + 1
                interval = interval.write(i, [start, end])
                return i + 1, interval

            def true_fn_last(interval):
                start = tf.where(x_target >= x[i])[0, 0]
                end = tf.where(x_target >= x[i])[-1, 0] + 1
                interval = interval.write(i, [start, end])
                return i + 1, interval

            def false_fn(interval):
                boolean_array = tf.logical_and(x_target >= x[i], x_target < x[i + 1])
                start = tf.where(boolean_array)[0, 0]
                end = tf.where(boolean_array)[-1, 0] + 1
                interval = interval.write(i, [start, end])
                return i + 1, interval

            i, interval = tf.cond(
                tf.equal(i, 0),
                lambda: true_fn_0(interval),
                lambda: tf.cond(tf.equal(i, self.n - 2), lambda: true_fn_last(interval), lambda: false_fn(interval))
            )

            return i, interval

        # Define the loop condition
        def loop_cond(i, interval):
            return i < self.n - 1

        # Initialize the loop variables
        i = tf.constant(0)

        # Run the loop
        _, interval = tf.while_loop(loop_cond, loop_body, [i, self.interval])

        # Convert the TensorArray to a Tensor
        self.interval = interval.stack()

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
