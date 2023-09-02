'''
A container for scipy/cupy data structures.
Currently Supports:
    Scipy for CPU
    Cupy for GPU
'''
import math
class SparseContainer:
    def __init__(self):
        self._np = None
        self._sparse = None
        self._linalg = None
        self.initialized = False

    def __call__(self, sparse_container):
        global np
        global sparse
        global linalg
        
        if sparse_container == 'scipy':
            import numpy as np
            import scipy.sparse as sparse
            import scipy.sparse.linalg as linalg
            
        elif sparse_container == 'cupy':
            import cupy as np
            import cupyx.scipy.sparse as sparse
            import cupyx.scipy.sparse.linalg as linalg
            
        else:
            raise ValueError(f"TuringLab::SolverFD::parse library must be one of: ['scipy', 'cupy']")

        self._np = np
        self._sparse = sparse
        self._linalg = linalg
        self.initialized = True
    
    def is_init(self):
        if not self.initialized:
            raise Exception("TuringLab::SolverFD unable to initialize. Exiting...")
    
    #using property decorator for is_init() check for elements.
    @property
    def np(self):
        self.is_init()
        return self._np

    @property
    def sparse(self):
        self.is_init()
        return self._sparse

    @property
    def linalg(self):
        self.is_init()
        return self._linalg

sparse_container = SparseContainer()

'''
A container for single axis over which a domain is discretized. Similar to a NumberLine :)

'''
class LinearAxisContainer:
    def __init__(self, name, **kwargs):
        self.name = name
        
        if all([arg in kwargs for arg in ['start', 'stop', 'num']]):
            start, stop, num = kwargs['start'], kwargs['stop'], kwargs['num']
            self.coords = sparse_container.np.linspace(start, stop, num)
            self.delta = abs(stop - start) / (num - 1)
            
        else:
            raise Exception('TuringLab::SolverFD::axis range must be supplied as a tuple (start, stop, num)')    
        self.num_points = len(self.coords)

'''
Sparse coefficient matrix container.
'''
class SparseMatrixContainer:

    def __init__(self, matrix):
        self.matrix = matrix

    #multiplication
    def __mul__(self, B):
        if isinstance(B, int) or isinstance(B, float):
            matrix = self.matrix * B
            return SparseMatrixContainer(matrix)
        
        elif isinstance(B, sparse_container.np.ndarray):
            if B.size == self.matrix.shape[0]:
                matrix = self.matrix.multiply(B.reshape(-1,1))
                return SparseMatrixContainer(matrix)
            else:
                raise ValueError(f"TuringLab::SolverFD::ndarray with shape {B.shape} is not compatible with SparseMatrixContainer with shape {self.matrix.shape}")
            
        else:
            return NotImplemented

    def __rmul__(self, B):
        return self * B
    
    #division
    def __truediv__(self, B):
        if isinstance(B, int) or isinstance(B, float):
            matrix = self.matrix / B
            return SparseMatrixContainer(matrix)
        
        elif isinstance(B, sparse_container.np.ndarray):
            if B.size == self.matrix.shape[0]:
                matrix = self.matrix.multiply(1 / B.reshape(-1,1))
                return SparseMatrixContainer(matrix)
            
            else:
                raise ValueError(f"TuringLab::SolverFD::ndarray with shape {B.shape} is not compatible with SparseMatrixContainer with shape {self.matrix.shape}")
        else:
            return NotImplemented
    
    def __rtruediv__(self, B):
        return NotImplemented

    #addition
    def __add__(self, B):
        if isinstance(B, SparseMatrixContainer):
            matrix = self.matrix + B.matrix
            return SparseMatrixContainer(matrix)
        else:
            return NotImplemented
    
    #subtraction
    def __sub__(self, B):
        if isinstance(B, SparseMatrixContainer):
            matrix = self.matrix - B.matrix
            return SparseMatrixContainer(matrix)
        else:
            return NotImplemented

    # negation
    def __neg__(self):
        return SparseMatrixContainer(-self.matrix)

'''
Contains a discritized representation of derivative. The context must be scaler.
'''        
class DiscretizedRepresentation:
    def __init__(self, matrix, scalar):
        
        self.matrix = matrix
        self.scalar = scalar
        self.model = scalar.model
        self.shape = scalar.shape

    def convert2matrix_container(self):
        scaler_idx = list(self.model.scalars.keys()).index(self.scalar.name)
        A_Matrix = sparse_container.sparse.csr_matrix((self.matrix.shape[0], sum(self.model.shape[:scaler_idx])))
        B_Matrix = sparse_container.sparse.csr_matrix((self.matrix.shape[0], sum(self.model.shape[scaler_idx + 1:])))
        model_matrix = sparse_container.sparse.hstack([A_Matrix, self.matrix, B_Matrix], format = 'csr')
        return SparseMatrixContainer(model_matrix)
        
    # slicing
    def __getitem__(self, slices):
        slice_index = sparse_container.np.zeros(self.shape, dtype = bool)
        slice_index[slices] = True
    
        matrix = self.matrix[slice_index.ravel()]
        return DiscretizedRepresentation(matrix, self.scalar).convert2matrix_container()

    # multiplication
    def __mul__(self, B):
        if isinstance(B, int) or isinstance(B, float):
            matrix = self.matrix * B
            return DiscretizedRepresentation(matrix, self.scalar)
        
        elif isinstance(B, sparse_container.np.ndarray):
            if B.shape == self.shape or (B.ndim == 1 and B.size == self.matrix.shape[0]):
                matrix = self.matrix.multiply(B.reshape(-1,1))
                return DiscretizedRepresentation(matrix, self.scalar)
            else:
                raise ValueError(f"TuringLab::SolverFD::ndarray with shape {B.shape} is not compatible with Scalar '{self.scalar.name}' with shape {self.shape}")

        elif isinstance(B, DiscretizedRepresentation):
            if self.scalar == B.scalar:
                matrix = self.matrix * B.matrix
                return DiscretizedRepresentation(matrix, self.scalar)
            else:
                raise NotImplementedError(f"TuringLab::SolverFD::DiscretizedScalars can only multiplied if they share the same Scalars. Got: '{self.scalar.name}' and '{B.scalar.name}'")
                
        else:
            return NotImplemented
        
    def __rmul__(self, B):
        return self * B

    # division
    def __truediv__(self, B):
        if isinstance(B, int) or isinstance(B, float):
            matrix = self.matrix / B
            return DiscretizedRepresentation(matrix, self.scalar)
        
        elif isinstance(B, sparse_container.np.ndarray):
            if B.shape == self.shape or (B.ndim == 1 and B.size == self.matrix.shape[0]):
                matrix = self.matrix.multiply(1 / B.reshape(-1,1))
                return DiscretizedRepresentation(matrix, self.scalar)
            else:
                raise ValueError(f"TuringLab::SolverFD::ndarray with shape {B.shape} is not compatible with Scalar '{self.scalar.name}' with shape {self.shape}")
        
        else:
            # inverse of a discritized scaler matrix is singular. Can't see use case.
            return NotImplemented

    def __rtruediv__(self, B):
        return NotImplemented
    
    # addition
    def __add__(self, B):
        if isinstance(B, DiscretizedRepresentation):
            if self.scalar == B.scalar:
                matrix = self.matrix + B.matrix
                return DiscretizedRepresentation(matrix, self.scalar)
            else:
                return self.convert2matrix_container() + B.convert2matrix_container()
            
        elif isinstance(B, SparseMatrixContainer):
            return self.convert2matrix_container() + B
        
        else:
            return NotImplemented

    def __radd__(self, B):
        if isinstance(B, SparseMatrixContainer):
            return B + self.convert2matrix_container()
        
        else:
            return NotImplemented
    
    # subtraction
    def __sub__(self, B):
        if isinstance(B, DiscretizedRepresentation):
            if self.scalar == B.scalar:
                matrix = self.matrix - B.matrix
                return DiscretizedRepresentation(matrix, self.scalar)
            else:
                return self.convert2matrix_container() - B.convert2matrix_container()
            
        elif isinstance(B, SparseMatrixContainer):
            return self.convert2matrix_container() - B
        
        else:
            return NotImplemented
    
    def __rsub__(self, B):
        if isinstance(B, SparseMatrixContainer):
            return B - self.convert2matrix_container()
        
        else:
            return NotImplemented

    # negation
    def __neg__(self):
        return DiscretizedRepresentation(-self.matrix, self.scalar)

def gen_coefficients(derivative, accuracy):
    n_forward = derivative + accuracy
    
    offsets = [sparse_container.np.arange(-i, n_forward-i, dtype = int) for i in range(n_forward)]
    
    c_accuracy = max(accuracy // 2 * 2, 2)
    n_central = max((derivative - 1) // 2 * 2 + 1 + c_accuracy, 3)
    
    if n_central < n_forward:
        offsets = offsets[:n_central//2] \
            + [sparse_container.np.arange(n_central, dtype = int) - n_central//2] \
            + offsets[-n_central//2 + 1:]

    coefficients = []
    for off in offsets:
        M_array = sparse_container.np.power(off, sparse_container.np.arange(len(off), dtype = int).reshape(-1,1))
        C_array = sparse_container.np.zeros(len(off))
        C_array[derivative] = math.factorial(derivative)
        coefficients.append(sparse_container.np.linalg.solve(M_array, C_array))
    return coefficients, offsets, n_forward

def gen_coeff_matrix(shape, dim_idx, derivative, accuracy):
    if dim_idx < 0:
        dim_idx = len(shape) + dim_idx
    if dim_idx < 0 or dim_idx >= len(shape):
        raise IndexError(f'TuringLab::SolverFD::dim_idx index out of range of shape')
        
    n_rows = shape[dim_idx]
    
    coefficients, offsets, n_forward = gen_coefficients(derivative, accuracy)
    
    if n_rows < n_forward:
        raise ValueError('TuringLab::SolverFD::Grid resolution is less than required')
        
    if n_rows > len(coefficients):
        central_idx = len(coefficients)//2
        
        forward = sparse_container.sparse.csr_matrix(sparse_container.np.array(coefficients[:central_idx]), shape = (central_idx, n_rows))
        
        central = sparse_container.sparse.diags(
            coefficients[central_idx],
            offsets[central_idx] + central_idx,
            shape=(n_rows - 2*central_idx , n_rows),
            format='csr')
        
        backwards = sparse_container.sparse.hstack([
            sparse_container.sparse.csr_matrix((central_idx, n_rows - n_forward)),
            sparse_container.sparse.csr_matrix(sparse_container.np.array(coefficients[-central_idx:]))
        ])
        
        fd_matrix = sparse_container.sparse.vstack([forward, central, backwards])
        
    else:
        fd_matrix = sparse_container.sparse.csr_matrix(sparse_container.np.stack(coefficients))

    A_matrix = sparse_container.sparse.identity(math.prod(shape[:dim_idx]))
    B_matrix = sparse_container.sparse.identity(math.prod(shape[dim_idx + 1:]))
    
    return sparse_container.sparse.kron(sparse_container.sparse.kron(A_matrix, fd_matrix), B_matrix, format = 'csr')

'''
Generates appropriately indexed coefficient matrices to build a finite difference simulation with.
'''
class ScalarMatrix:
    def __init__(self, name, axes, accuracy = 2):
        self.name = name
        self.axes = {}
        for a in axes:
            if a.name in self.axes:
                raise ValueError(f'TuringLab::SolverFD::Axis {a.name} is duplicated in Scalar {name}')
            self.axes.update({a.name:a})
        
        self.accuracy = accuracy
        
        self.shape = tuple(a.num_points for a in self.axes.values())
        self.size = math.prod(self.shape)
        
        self.model = None
        self.identity = sparse_container.sparse.identity(self.size, format='csr')
        self.coords = sparse_container.np.meshgrid(*tuple([a.coords for a in self.axes.values()]), indexing = 'ij')
        self.coords = {axis:coord for axis, coord in zip(self.axes.keys(), self.coords)}
        
        self.timestep = None
    
    def _check_model(self):
        if self.model is None:
            raise Exception(f'TuringLab::SolverFD::ScalarMatrix {self.name} has not been assigned to a model')
    
    @property
    def i(self):
        self._check_model()
        return DiscretizedRepresentation(self.identity, self)
    
    def d(self, axis_name, derivative = 1, accuracy = None):
        self._check_model()
            
        if accuracy is None:
            accuracy = self.accuracy

        if accuracy < 1:
            raise ValueError('TuringLab::SolverFD::Derivative approximation accuracy must be at least 1')
            
        if derivative < 1:
            raise ValueError("TuringLab::SolverFD::Derivative must be at least 1. Were you looking for 'Scalar.i' instead?") 
        
        if axis_name in self.axes.keys(): 
            dim_idx = list(self.axes.keys()).index(axis_name)
        else:
            raise ValueError(f'TuringLab::SolverFD::Derivative axis {axis_name} is not registered: {list(self.axes.keys())}')
                
        coeff_matrix = gen_coeff_matrix(self.shape, dim_idx, derivative, accuracy).copy()
        coeff_matrix /= self.axes[axis_name].delta**derivative
        return DiscretizedRepresentation(coeff_matrix, self)
        
    def dt(self, dt_type, derivative = 1, accuracy = None):
        self._check_model()
            
        if accuracy is None:
            accuracy = self.accuracy
            
        if accuracy < 1:
            raise ValueError('TuringLab::SolverFD::Derivative approximation accuracy must be at least 1')
            
        if derivative < 1:
            raise ValueError("TuringLab::SolverFD::Derivative must be at least 1. Were you looking for 'Scalar.i' instead?") 
        
        if self.timestep is None:
            raise ValueError('Timestep is not set. Create model with FiniteDifferenceSolver(scalars = [?], timestep = ?)')
        
        coefficients, *_ = gen_coefficients(derivative, accuracy)
        coefficients = coefficients[-1] / self.timestep**derivative
    
        if dt_type == 'lhs':
            return self.i * float(coefficients[-1])
        elif dt_type == 'rhs':
            return sparse_container.sparse.kron(sparse_container.sparse.identity(self.size), sparse_container.sparse.csr_matrix(coefficients[:-1]), format = 'csr')
        else:
            raise ValueError(f"TuringLab::SolverFD::dt_type must be either 'scalar' or 'constraint' for {self.name}. Got {dt_type}.")

'''
This is the main datastructure aka finite difference solver.
'''
class FiniteDifferenceSolver:
    def __init__(self, scalars, timestep = None):
        
        self.scalars = {}
        for s in scalars:
            if s.name in self.scalars:
                raise ValueError(f'TuringLab::SolverFD::Scalar {s.name} is duplicated')
            self.scalars.update({s.name:s})
                
        self.shape = tuple(s.size for s in self.scalars.values())
        self.size = sum(self.shape)
        self.timestep = timestep
    
        for s in self.scalars.values():
            s.model = self
            if self.timestep is not None:
                s.timestep = self.timestep
        
        self.coords = {key:scalar.coords for key, scalar in self.scalars.items()} 

        self.equations = {}
        self.bocos = {}
        
        self.equation_coefficients_built = False
        self.equation_constraints_built = False
        self.boco_coefficients_applied = False
        self.boco_coefficients_applied = False

    def update_equations(self, equations, purge = False):
        if purge:
            self.equations = {}
            self.equation_coefficients_built = False
            self.equation_constraints_built = False

        for key, (coeff, const) in equations.items():
            old_coeff, old_const = self.equations.get(key, (None, None))

            if isinstance(coeff, SparseMatrixContainer):
                new_coeff = coeff.matrix
                self.equation_coefficients_built = False
            elif isinstance(coeff, DiscretizedRepresentation):
                new_coeff = coeff.convert2matrix_container().matrix
                self.equation_coefficients_built = False
            elif coeff is None:
                new_coeff = old_coeff
            else:
                raise TypeError('TuringLab::SolverFD::Matrix coefficients must be [SparseMatrixContainer, DiscritizedScalar, or None]')

            if isinstance(const, sparse_container.np.ndarray):
                new_const = const
                self.equation_constraints_built = False
            elif isinstance(const, float) or isinstance(const, int):
                new_const = sparse_container.np.ones(new_coeff.shape[0]) * const
                self.equation_constraints_built = False
            elif const is None:
                new_const = old_const
            else:
                raise TypeError('TuringLab::SolverFD::Matrix constraints must be [ndarray, float, int, or None]')
        
            self.equations.update({key: (new_coeff, new_const)})

        for key, (mask, coeff, const, coeff_applied, const_applied) in self.bocos.items():
            if not self.equation_coefficients_built:
                coeff_applied = False
            if not self.equation_constraints_built:
                const_applied = False
            self.bocos.update({key: (mask, coeff, const, coeff_applied, const_applied)})
    
    def update_bocos(self, bocos, purge = False):
        if purge:
            self.bocos = {}
            self.boco_coefficients_applied = False
            self.boco_constraints_applied = False
            
        for key, (mask, coeff, const) in bocos.items():
            old_mask, old_coeff, old_const, coeff_applied, const_applied = self.bocos.get(key, (None, None, None, False, False))
            if isinstance(mask, SparseMatrixContainer) or isinstance(mask, DiscretizedRepresentation):
                boco_mask = mask.matrix.T if isinstance(mask, SparseMatrixContainer) else mask.convert2matrix_container().matrix.T
                vec_mask = sparse_container.np.squeeze(sparse_container.np.array(boco_mask.sum(axis = 1)))
                coeff_mask = sparse_container.sparse.diags(1 - vec_mask)
                
                new_mask = (coeff_mask, boco_mask, vec_mask.astype(bool))
                
                self.boco_coefficients_applied = False
                self.boco_constraints_applied = False
                coeff_applied = False
                const_applied = False
            
            elif mask is None:
                new_mask = old_mask
            else:
                raise TypeError("TuringLab::SolverFD::Boundary condition mask for '{key}' must be [SparseMatrixContainer, DiscritizedScalar, or None]")

            if isinstance(coeff, SparseMatrixContainer) or isinstance(coeff, DiscretizedRepresentation):
                
                new_coeff = coeff.matrix if isinstance(coeff, SparseMatrixContainer) else coeff.convert2matrix_container().matrix
                self.boco_coefficients_applied = False
                coeff_applied = False
            
            elif coeff is None:
                new_coeff = old_coeff
            
            else:
                raise TypeError("TuringLab::SolverFD::Boundary condition coefficients for '{key}' must be [SparseMatrixContainer, DiscritizedScalar, or None]")

            if isinstance(const, sparse_container.np.ndarray):
                new_const = const.ravel() if const.ndim > 1 else const
                self.boco_constraints_applied = False
                const_applied = False
            
            elif isinstance(const, float) or isinstance(const, int):
                new_const = sparse_container.np.ones(new_coeff.shape[0]) * const
                self.boco_constraints_applied = False
                const_applied = False
            elif const is None:
                new_const = old_const
            else:
                raise TypeError("TuringLab::SolverFD::Boundary condition constraints for '{key}' must be [ndarray, float, int, or None]")
            self.bocos.update({key: (new_mask, new_coeff, new_const, coeff_applied, const_applied)})
    
    def check_equation(self, key, coeff, const, check_type = 'equation'):
        if coeff is None:
            raise Exception(f"TuringLab::SolverFD::coefficient matrix for {check_type} '{key}' has not been specified")
        if const is None:
            raise Exception(f"TuringLab::SolverFD::Constraint vector matrix for {check_type} '{key}' has not been specified")
        if coeff.shape[0] != const.size:
            raise Exception(f"TuringLab::SolverFD::Number of rows in {check_type} '{key}' coefficient matrix must equal constraint vector length. Got shapes: {coeff.shape}, {const.shape}")

    def check_boco(self, key, mask, coeff, const):
        if mask is None:
            raise Exception(f"TuringLab::SolverFD::Mask for boundary condition '{key}' has not been specified")
        self.check_equation(key, coeff, const, check_type = 'boundary condition')

    def build(self):
        if not self.equation_coefficients_built:
            eq_coefficients = []
            for key, (coeff, const) in self.equations.items():
                self.check_equation(key, coeff, const)                
                eq_coefficients.append(coeff)
                
            self._coefficients = sparse_container.sparse.vstack(eq_coefficients, format = 'csr')
        
            if self._coefficients.shape[0] < self.size:
                raise Exception(f"TuringLab::SolverFD::Solution underspecified. Got {self._coefficients.shape[0]} equations and {self.size} unknowns")
            if self._coefficients.shape[0] > self.size:
                raise Exception(f"TuringLab::SolverFD::Solution overspecified. Got {self._coefficients.shape[0]} equations and {self.size} unknowns")
            
            self.equation_coefficients_built = True
            self.boco_coefficients_applied = False

        if not self.equation_constraints_built:
            eq_constraints = []
            for key, (coeff, const) in self.equations.items():
                self.check_equation(key, coeff, const)       
                eq_constraints.append(const)
                
            self._constraints = sparse_container.np.hstack(eq_constraints)
            
            self.equation_constraints_built = True
            self.boco_constraints_applied = False
            
        if not self.boco_coefficients_applied:
            self.coefficients = self._coefficients.copy()
            for key, (mask, coeff, const, coeff_applied, const_applied) in self.bocos.items():
                if coeff_applied: continue
                self.check_boco(key, mask, coeff, const)
                
                coeff_mask, boco_mask, vec_mask = mask
                self.coefficients = coeff_mask * self.coefficients + boco_mask * coeff
                
                coeff_applied = True
                self.bocos.update({key: (mask, coeff, const, coeff_applied, const_applied)})
        
            self.boco_coefficients_applied = True

        if not self.boco_constraints_applied:
            self.constraints = self._constraints.copy()
            for key, (mask, coeff, const, coeff_applied, const_applied) in self.bocos.items():
                if const_applied: continue
                self.check_boco(key, mask, coeff, const)
                
                coeff_mask, boco_mask, vec_mask = mask
                self.constraints[vec_mask] = const
                
                const_applied = True
                self.bocos.update({key: (mask, coeff, const, coeff_applied, const_applied)})
                
            self.boco_constraints_applied = True

    '''
    Solve and return the solution.
    '''
    def solve(self, solver = 'spsolve'):
        
        self.build()
        if solver == 'spsolve':
            soln = sparse_container.linalg.spsolve(self.coefficients, self.constraints)
        elif solver == 'lsqr':
            soln = sparse_container.linalg.lsqr(self.coefficients, self.constraints)[0]

        output = {}
        i = 0
        for key, scalar in self.scalars.items():
            output.update({
                key:
                soln[i:i + scalar.size].reshape(scalar.shape)
            })
            i += scalar.size
        return output