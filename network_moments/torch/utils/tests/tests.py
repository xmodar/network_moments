import torch
from ..ops import sqrtm
from ..rand import definite
from unittest import TestCase
from torch.autograd import gradcheck


class MatrixSquareRootTest(TestCase):
    '''Test sqrtm and MatrixSquareRoot.'''

    seed = None
    dims = 20
    sigma = 10
    dtype = torch.float64
    device = torch.device('cpu')

    def setUp(self):
        cls = type(self)
        if cls.seed is not None:
            torch.manual_seed(cls.seed)
        self.x = definite(cls.dims, norm=cls.sigma ** 2,
                          dtype=cls.dtype, device=cls.device)

    def tearDown(self):
        del self.x

    def test_forward(self):
        s = sqrtm(self.x)
        y = s.mm(s)
        self.assertTrue(
            torch.allclose(y, self.x),
            ((self.x - y).norm() / self.x.norm()).item()
        )

    def test_backward(self):
        msg = ''
        try:
            x = torch.autograd.Variable(self.x, requires_grad=True)
            gradcheck(sqrtm, (x,), rtol=1e-2, atol=1 / type(self).sigma)
        except RuntimeError as exc:
            msg = str(exc)
        if msg != '':
            self.fail(msg)
