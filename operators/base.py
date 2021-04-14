import torch

__all__ = [
    "Chain",
    "dottest",
    "Hessian",
]


class Chain(torch.nn.Module):
    
    def __init__(self, ops: list):
        super(Chain, self).__init__()
        assert len(ops) >= 1
        self.ops = ops
        
    def forward(self, x):
        if len(self.ops) == 1:
            return self.ops[0](x)
        else:
            out = self.ops[0](x)
            for Op in self.ops[1:]:
                out = Op(out)
            return out
    
    def adjoint(self, x):
        if len(self.ops) == 1:
            return self.ops[-1].adjoint(x)

        else:
            out = self.ops[-1].adjoint(x)
            for Op in self.ops[::-1][1:]:
                out = Op.adjoint(out)
        return out
    
    def __getitem__(self, item):
        return self.ops[item]


class Hessian(torch.nn.Module):
    
    def __init__(self, op):
        super(Hessian, self).__init__()
        self.op = op
    
    def forward(self, x):
        return self.op.adjoint(self.op.forward(x))
    
    def adjoint(self, x):
        return self.forward(x)
    

def dottest(op, domain_tensor, range_tensor):
    d1 = torch.randn(domain_tensor.shape)
    r1 = torch.randn(range_tensor.shape)
    
    r2 = op.forward(d1)
    d2 = op.adjoint(r1)
    
    d_ = torch.vdot(d1.view(-1), d2.view(-1))
    r_ = torch.vdot(r1.view(-1), r2.view(-1))
    
    err_abs = d_ - r_
    err_rel = err_abs / d_
    
    print("Absolute error: %.6e" % abs(err_abs.item()))
    print("Relative error: %.6e \n" % abs(err_rel.item()))

