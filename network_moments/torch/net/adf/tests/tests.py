import torch
from torch import nn
from itertools import product
from unittest import TestCase
from ....utils import rand, map_batch
from ....utils import cov as compute_cov
from ....gaussian.relu import batch_moments as relu_moments
from ....general.affine import batch_moments as affine_moments


__all__ = ['FunctionTestCase', 'MomentsTestCase',
           'GaussianAffineMomentsTest', 'GaussianReluMomentsTest']


class FunctionTestCase(type):
    '''Adopts unittest.TestCase for general functions.

    The instance classes of this metaclass should implement
    self.execute(*args) function and should contain an iterable
    called all_tests of args tuples to self.execute().

    Example:
    def my_func(a, b, c):
        return (a + b) * c
    class MyFuncTest(metaclass=FunctionTestCase):
        func = my_func
        all_tests = [  # consider using collections.product
            [(1, 2, 3), 9],
            [(1, 3, 3), 12],
            [(2, 1, 3), 9],  # you can pass whatever
            [(3, 3, 3), 18],
        ]
        def setUp(self):
            pass
        def tearDown(self):
            pass
        def execute(self, inputs, expected):
            a, b, c = inputs
            out = type(self).func(a, b, c)
            msg = '({} + {}) * {} = {} != {}'
            msg = msg.format(a, b, c, expected, out)
            self.assertEqual(out, expected, msg)
    '''
    def __new__(cls, name, bases, dic):
        if TestCase not in bases:
            bases = bases + (TestCase,)
        for i, args in enumerate(dic['all_tests'], 1):
            def func(self, args=args):
                return self.execute(*args)
            if 'get_description' in dic:
                func.__doc__ = dic['get_description'].__func__(*args)
            dic['test_{}'.format(i)] = func
        clas = super().__new__(cls, name, bases, dic)
        return clas


class MomentsTestCase(metaclass=FunctionTestCase):
    all_tests = []         # No tests for this base class

    mbatch = 2             # Batch size of the input mean.
    vbatch = 2             # Batch size of the input covariance.
    length = 3             # Size of the vector.
    factor = 10            # To multiply the mean and standard deviation.
    count = 1000000        # Number of samples for Monte-Carlo estimation.
    seed = None            # Seed for the random number generator.
    device = None          # In which device.
    dtype = torch.float64  # The data type.

    def setUp(self):
        cls = type(self)
        if cls.seed is not None:
            torch.manual_seed(cls.seed)

        # input mean and covariance
        self.mu = torch.randn(cls.mbatch, cls.length, dtype=cls.dtype,
                              device=cls.device) * cls.factor
        self.cov = rand.definite(cls.length, batch=cls.vbatch,
                                 dtype=cls.dtype, device=cls.device,
                                 positive=True, semi=False,
                                 norm=cls.factor**2)
        self.var = map_batch(lambda x: x.diag(), self.cov)

    def tearDown(self):
        del (self.mu, self.cov, self.var)

    def execute(self, *args):
        desc, arguments, expect = self.get_io(*args)

        outputs = type(self).func(**arguments)
        if not isinstance(outputs, tuple):
            outputs = (outputs,)

        msg = 'Got {} outputs while expecting {}'
        self.assertEqual(len(outputs), len(expect),
                         msg.format(len(outputs), len(expect)))

        passed_all = True
        msg = ('\nExpected output #{}:\n{}\n'
               '!#!-Got:\n{}\n'
               '!#!-min_abs_err    = {}\n'
               '!#!-mean_abs_err   = {}\n'
               '!#!-max_abs_err    = {}\n'
               '!#!-normalized_err = {}\n')
        for i, (o, e) in enumerate(zip(outputs, expect)):
            cond = o.size() == e.size() and torch.allclose(o, e, rtol=1e-1)
            if not cond:
                passed_all = False
                err = (e.data - o.data)
                abs_err = err.abs()
                desc += msg.format(
                    i, e.data.cpu().numpy(), o.data.cpu().numpy(),
                    abs_err.min().item(),
                    abs_err.mean().item(),
                    abs_err.max().item(),
                    (err.norm() / e.data.norm()).item(),
                )
            else:
                desc += '\nOutput #{} passed'.format(i)
        self.assertTrue(passed_all, desc)
        return tuple(zip(expect, outputs))


class GaussianAffineMomentsTest(MomentsTestCase):
    '''Test the tightness against Monte-Carlo estimations.

    The expressions are for the affine transformation f(x) = A * x + b.
    '''
    func = affine_moments

    all_tests = (x for x in product(
        ('weight', 'forward', 'linearize'),      # 0: func
        ('variance', 'diagonal', 'covariance'),  # 1: in_var
        (True, False),                           # 2: mean
        ('None', 'variance', 'covariance'),      # 3: out_var
    ) if (x[2] or x[3] != 'None'))

    @staticmethod
    def get_description(func, in_var, mean, out_var):
        m = 'mean' if mean else False
        v = False if out_var == 'None' else out_var
        return 'IN{} OUT{}'.format([func, in_var], [x for x in (m, v) if x])

    def setUp(self):
        super().setUp()
        cls = type(self)
        self.f = nn.Linear(cls.length, cls.length).to(cls.device, cls.dtype)
        self.F = lambda x: self.f(x)  # hides the weights

    def tearDown(self):
        super().tearDown()
        del (self.f, self.F)

    def get_io(self, func, in_var, mean, out_var):
        # get test description
        if out_var == 'None' and not mean:
            raise self.skipTest('No output case')
        desc = self.get_description(func, in_var, mean, out_var)

        # get test arguments
        args = {
            'mu': self.mu,
            'jacobian': func == 'linearize',
            'mean': mean,
            'diagonal': out_var == 'variance',
        }
        if func == 'weight':
            args['f'] = self.f
        elif func == 'forward' or args['jacobian']:
            args['f'] = self.F
        else:
            raise ValueError('func not in {weight, linearize, forward}')
        if in_var == 'variance':
            args['var'] = self.var
        elif in_var == 'covariance':
            args['var'] = self.cov
        elif in_var == 'diagonal':
            args['var'] = map_batch(lambda x: x.diag(), self.var)
        else:
            raise ValueError('in_var not in {variance, covariance, diagonal}')
        covariance = variance = False
        if out_var == 'None':
            args['covariance'] = False
        elif out_var == 'covariance':
            args['covariance'] = covariance = True
        elif args['diagonal']:
            args['covariance'] = variance = True
        else:
            raise ValueError('out_var not in {variance, covariance, None}')

        # get test expected outputs
        def expected(m, v):
            if v.dim() != 2:
                v = v.diag()
            normal = torch.distributions.MultivariateNormal(m, v)
            out_samples = self.f(normal.sample((type(self).count,)))
            mu = torch.mean(out_samples, dim=0) if args['mean'] else None
            var = torch.var(out_samples, dim=0) if variance else None
            cov = compute_cov(out_samples) if covariance else None
            return tuple(r for r in (mu, var, cov) if r is not None)

        expect = map_batch(expected, args['mu'], args['var'])
        if func == 'weight' and out_var != 'None' and args['var'].size(0) == 1:
            expect = (*expect[:-1], expect[-1][:1, ...])
        return desc, args, expect


class GaussianReluMomentsTest(MomentsTestCase):
    '''Test the tightness against Monte-Carlo estimations.

    The expressions are for the affine transformation f(x) = A * x + b.
    '''
    func = relu_moments

    all_tests = (x for x in product(
        ('variance', 'diagonal', 'covariance'),  # 0: in_var
        (True, False),                           # 1: mean
        ('None', 'variance', 'covariance'),      # 2: out_var
    ) if (x[1] or x[2] != 'None'))

    @staticmethod
    def get_description(in_var, mean, out_var):
        m = 'mean' if mean else False
        v = False if out_var == 'None' else out_var
        return 'IN{} OUT{}'.format([in_var], [x for x in (m, v) if x])

    def get_io(self, in_var, mean, out_var):
        # get test description
        if out_var == 'None' and not mean:
            raise self.skipTest('No output case')
        desc = self.get_description(in_var, mean, out_var)

        # get test arguments
        args = {
            'mu': self.mu * 0,
            'mean': mean,
            'diagonal': out_var == 'variance',
        }

        if in_var == 'variance':
            args['var'] = self.var
        elif in_var == 'covariance':
            args['var'] = self.cov
        elif in_var == 'diagonal':
            args['var'] = map_batch(lambda x: x.diag(), self.var)
        else:
            raise ValueError('in_var not in {variance, covariance, diagonal}')
        covariance = variance = False
        if out_var == 'None':
            args['covariance'] = False
        elif out_var == 'covariance':
            args['covariance'] = covariance = True
        elif args['diagonal']:
            args['covariance'] = variance = True
        else:
            raise ValueError('out_var not in {variance, covariance, None}')

        # get test expected outputs
        def expected(m, v):
            if v.dim() != 2:
                v = v.diag()
            normal = torch.distributions.MultivariateNormal(m, v)
            samples = normal.sample((type(self).count,)).clamp(min=0)
            mu = torch.mean(samples, dim=0) if args['mean'] else None
            var = torch.var(samples, dim=0) if variance else None
            cov = compute_cov(samples) if covariance else None
            return tuple(r for r in (mu, var, cov) if r is not None)

        expect = map_batch(expected, args['mu'], args['var'])
        if out_var != 'None' and args['var'].size(0) == 1:
            expect = (*expect[:-1], expect[-1][:1, ...])
        return desc, args, expect
