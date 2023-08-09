import unittest
import numpy as np
from timeseries.models import HeisenbergBonded
from timeseries.models._heisenberg import HeisenbergXYZ
from timeseries.utils import int_to_bitstring, bitstring_to_circuit
from pytket.extensions.qiskit import AerStateBackend
from pytket.circuit import Circuit
from parameterized import parameterized

Jxy = -1
Jz = -2
hz = 3.0

class TestHeisenbergBonded(unittest.TestCase):

    # def __init__(self, *args, **kwargs):
    #     super(TestHeisenberg, self).__init__(*args, **kwargs)
    @classmethod
    def setUpClass(cls):
        L = 9
        coupling_list = [[0,1], [3,4], [6,7], 
                        [1,2], [4,5], [7,8], 
                        [0,3], [1,4], [2,5], 
                        [3,6], [4,7], [5,8]]

        n_steps = 5
        dt = 0.1
        random_seed = 112

        model = HeisenbergBonded.from_coupling_list(L, coupling_list, Jxy, Jxy, Jz, hz, False)
        # Initialize Psi to a random vector in the computational basis
        np.random.seed(random_seed)
        state_index = np.random.randint(0, 2**L)
        psi_0 = np.zeros(2**L)
        psi_0[state_index] = 1.0
        psi_T_expected_1st = model.evolve_state_1st_trotter(psi_0, n_steps*dt, n_steps)
        G_T_expected_1st = np.vdot(psi_0, psi_T_expected_1st)        
        psi_T_expected_2nd = model.time_evolution_2nd_trotter(psi_0, n_steps*dt, n_steps)
        G_T_expected_2nd = np.vdot(psi_0, psi_T_expected_2nd)

        bitstring = int_to_bitstring(state_index, L)
        psi_0_circuit = bitstring_to_circuit(bitstring)


        cls.backend = AerStateBackend()
        cls.model = model
        cls.L = L
        cls.n_steps = n_steps
        cls.dt = dt
        cls.psi_0_circuit = psi_0_circuit
        cls.psi_0 = psi_0
        cls.psi_bitstring = bitstring
        cls.psi_T_expected_1st = psi_T_expected_1st
        cls.G_T_expected_1st = G_T_expected_1st        
        cls.psi_T_expected_2nd = psi_T_expected_2nd
        cls.G_T_expected_2nd = G_T_expected_2nd

    def test_exact_time_evolution(self):
        """Test the time evolution of XXZ model by comparing it with XYZ model."""
        L = 9
        coupling_list = [[0,1], [3,4], [6,7], [1,2], [4, 5], [7, 8], 
                        [0,3], [1,4], [2,5], [3,6], [4,7], [5,8]]
        order = 'xyz'
        T = 10

        model2 = HeisenbergXYZ.from_coupling_list(L, coupling_list, Jxy, Jxy, Jz, hz, order)
        psi_T_actual = self.model.evolve_state_exact(self.psi_0, T)
        psi_T_expected = model2.evolve_state_exact(self.psi_0, T)
        np.testing.assert_almost_equal(psi_T_actual, psi_T_expected , decimal=7, err_msg='', verbose=True) 

    def test_symmetry(self):
        """Test the Trotterization of XXZ model conserves U(1) symmetry."""
        dt = 0.1
        n_steps = 100

        psi_0 = np.zeros(2**self.L)
        psi_0[0] = 1.0

        psi_T_actual = self.model.evolve_state_1st_trotter(psi_0, n_steps*dt, n_steps)
        np.testing.assert_equal(psi_T_actual[1:], 0 , err_msg='First-order Trotterizaiton does not conserve symmetry.', verbose=True) 
        psi_T_actual = self.model.evolve_state_2nd_trotter(psi_0, n_steps*dt, n_steps)
        np.testing.assert_equal(psi_T_actual[1:], 0 , err_msg='Second-order Trotterizaiton does not conserve symmetry.', verbose=True) 
        
    def test_1st_trotter_circuit_without_control(self):
        """Test the 1st-order Trotterized circuit of XXZ model without control qbit."""

        qc = Circuit(self.L)
        qc.append(self.psi_0_circuit)        
        qc.append(self.model.make_1st_trotter_circuit(self.n_steps, self.dt, 0))
        compiled_circ = self.backend.get_compiled_circuit(qc)
        psi_T_actual = self.backend.run_circuit(compiled_circ).get_state()
        np.testing.assert_almost_equal(psi_T_actual, self.psi_T_expected_1st , decimal=7, err_msg='', verbose=True)

    def test_1st_trotter_circuits_with_control(self):
        """Test the 1st-order Trotterized circuit of XYZ model with a control qbit."""

        qc = Circuit(self.L+1)
        qc.add_circuit(self.psi_0_circuit, list(range(1, self.L+1)))
        qc.X(0) #Set control qubit to 1
        qc.append(self.model.make_1st_trotter_circuit(self.n_steps, self.dt, 1))

        compiled_circ = self.backend.get_compiled_circuit(qc)
        psi_T_actual = self.backend.run_circuit(compiled_circ).get_state()[2**self.L:] #The first-half of state vector is zero
        
        np.testing.assert_almost_equal(psi_T_actual, self.psi_T_expected_1st , decimal=7, err_msg='', verbose=True)


    def test_2nd_trotter_circuit_without_control(self):
        """Test the 2nd-order Trotterized circuit of XXZ model without control qbit."""

        qc = Circuit(self.L)
        qc.append(self.psi_0_circuit)        
        qc.append(self.model.make_2nd_trotter_circuit(self.n_steps, self.dt, 0))
        compiled_circ = self.backend.get_compiled_circuit(qc)
        psi_T_actual = self.backend.run_circuit(compiled_circ).get_state()
        np.testing.assert_almost_equal(psi_T_actual, self.psi_T_expected_2nd , decimal=7, err_msg='', verbose=True)
        
    def test_2nd_trotter_circuits_with_control(self):
        """Test the 2nd-order Trotterized circuit of XYZ model with a control qbit."""

        qc = Circuit(self.L+1)
        qc.add_circuit(self.psi_0_circuit, list(range(1, self.L+1)))
        qc.X(0) #Set control qubit to 1
        qc.append(self.model.make_2nd_trotter_circuit(self.n_steps, self.dt, 1))

        compiled_circ = self.backend.get_compiled_circuit(qc)
        psi_T_actual = self.backend.run_circuit(compiled_circ).get_state()[2**self.L:] #The first-half of state vector is zero
        
        np.testing.assert_almost_equal(psi_T_actual, self.psi_T_expected_2nd , decimal=7, err_msg='', verbose=True)

    @parameterized.expand([
        [1, False, True, 1],    
    ])
    def test_hadamard_circuit_real(self, trotter_order, split, optimized, n_controls):
        """Test the real-part Hadamard test circuit for XXZ model."""
        qc = self.model.make_hadamard_circuit(self.psi_0_circuit, self.dt*self.n_steps, self.n_steps, trotter_order, split, optimized, n_controls, False)
        compiled_circ = self.backend.get_compiled_circuit(qc)       
        full_psi = self.backend.run_circuit(compiled_circ).get_state()
        p_ancilla_0 = np.linalg.norm(full_psi[:2**self.L])**2
        p_ancilla_1 = np.linalg.norm(full_psi[2**self.L:])**2
        G_T_actual_real = p_ancilla_0-p_ancilla_1

        p_zero_state = (abs(full_psi[0])**2+abs(full_psi[-2**self.L])**2)
        G_T_actual_norm2 = 2*p_zero_state-1

        np.testing.assert_almost_equal(G_T_actual_real,  self.G_T_expected(trotter_order).real, decimal=7, err_msg=', real parts of G(t) mismatch', verbose=True)
        np.testing.assert_almost_equal(G_T_actual_norm2,  abs(self.G_T_expected(trotter_order))**2, decimal=7, err_msg=', magnitudes of G(t) mismatch', verbose=True)

    def G_T_expected(self, trotter_order):
        if(trotter_order == 1):
            return self.G_T_expected_1st
        elif(trotter_order == 2):
            return self.G_T_expected_2nd
        else:
            raise ValueError("Only up to 2nd-order Trotterization is available")

    @parameterized.expand([
        [1],
        [2],      
    ])
    def test_amplitude_circuit(self, trotter_order):
        """Test the amplitude circuit for XXZ model."""
        qc = self.model.make_amplitude_circuit(self.psi_0_circuit, self.dt*self.n_steps, self.n_steps, trotter_order)
        compiled_circ = self.backend.get_compiled_circuit(qc)
        new_psi = self.backend.run_circuit(compiled_circ).get_state()
        G_T_actual_norm2 = np.linalg.norm(new_psi[0])**2
        np.testing.assert_almost_equal(G_T_actual_norm2,  abs(self.G_T_expected(trotter_order))**2, decimal=7, err_msg=', magnitudes of G(t) mismatch', verbose=True)

    @parameterized.expand([
        [False, 1, True, 0],
        [False, 1, False, 0],
        [False, 2, True, 0],
        [False, 2, False, 0],    
        [False, 2, False, 100],
        [True, 2, False, 100],
    ])
    def test_cat_circuit(self, imagninary, trotter_order, log_preperation, energy_shift):
        """Test that Loschmidt echo is estimated correctly using cat state circuits for XXZ model."""
        T = self.dt*self.n_steps

        qc_real = self.model.make_cat_circuit(self.psi_bitstring, T, self.n_steps, trotter_order, imagninary, energy_shift, log_preperation)
        compiled_circ_real = self.backend.get_compiled_circuit(qc_real)
        state_real = self.backend.run_circuit(compiled_circ_real).get_state()
        p_0 = abs(state_real[0])**2
        
        control_qbit = list(self.psi_bitstring).index(1)
        pi_state_index = 2**(self.L-1-control_qbit)
        p_pi = abs(state_real[pi_state_index])**2

        G_T_actual_part = (p_0-p_pi)
        G_T_actual_norm2 = 2*(p_0+p_pi)-1

        G_T_expected = self.G_T_expected(trotter_order)*np.exp(1j*energy_shift*T)

        if(imagninary):
            np.testing.assert_almost_equal(G_T_actual_part,  G_T_expected.imag, decimal=7, err_msg=', imagninary part of G(t) mismatch', verbose=True)
        else:
            np.testing.assert_almost_equal(G_T_actual_part,  G_T_expected.real, decimal=7, err_msg=', real part of G(t) mismatch', verbose=True)
        np.testing.assert_almost_equal(G_T_actual_norm2,  abs(G_T_expected)**2, decimal=7, err_msg=', magnitude of G(t) mismatch', verbose=True)

if __name__ == '__main__':
    unittest.main()
