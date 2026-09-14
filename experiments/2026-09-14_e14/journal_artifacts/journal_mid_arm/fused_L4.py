import flopscope.numpy as fnp
class Estimator:
    def setup(self, context): pass
    def predict(self, mlp, budget): return fnp.zeros((len(mlp.weights), mlp.width), dtype=fnp.float32)
