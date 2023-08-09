import unittest
import numpy as np
from timeseries.models._heisenberg import HeisenbergXYZ
from timeseries.utils import int_to_bitstring, bitstring_to_circuit
from pytket.extensions.qiskit import AerStateBackend
from pytket.circuit import Circuit
from parameterized import parameterized


class TestHeisenbergXYZ(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        L = 9
        coupling_list = [[0,1], [3,4], [6,7], [1,2], [4, 5], [7, 8], 
                        [0,3], [1,4], [2,5], [3,6], [4,7], [5,8]]
        Jx = -1
        Jy = -2
        Jz = -3
        hz = 4.0
        order = 'xyz'
        n_steps = 3
        dt = 1.0
        random_seed = 100

        model = HeisenbergXYZ.from_coupling_list(L, coupling_list, Jx, Jy, Jz, hz, order)
        # Initialize Psi to a random vector in the computational basis
        np.random.seed(random_seed)
        state_index = np.random.randint(0, 2**L)
        psi_0 = np.zeros(2**L)
        psi_0[state_index] = 1.0
        psi_T_expected_1st = model.evolve_state_1st_trotter(psi_0, n_steps*dt, n_steps)
        G_T_expected_1st = np.vdot(psi_0, psi_T_expected_1st)        
        psi_T_expected_2nd = model.evolve_state_2nd_trotter(psi_0, n_steps*dt, n_steps)
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
        cls.psi_T_expected_1st = psi_T_expected_1st
        cls.G_T_expected_1st = G_T_expected_1st        
        cls.psi_T_expected_2nd = psi_T_expected_2nd
        cls.G_T_expected_2nd = G_T_expected_2nd

    def test_1st_trotter_circuit_without_control(self):
        """Test the 1st-order Trotterized circuit of XYZ model without control qbit."""

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
        """Test the 2nd-order Trotterized circuit of XYZ model without control qbit."""

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

    def test_split_2nd_trotter_circuits_with_control(self):
        """Test the splitted 2nd-order Trotterized circuits of TFIM with a control qbit."""
     
        qc = Circuit(self.L+1)
        state_qbits = list(range(1, self.L+1))
        qc.add_circuit(self.psi_0_circuit, state_qbits)
        qc.X(0) #Set control qubit to 1
        main, residue = self.model.make_split_2nd_trotter_circuits(self.n_steps, self.dt, 1)
        qc.add_circuit(residue, state_qbits)
        qc.append(main)
        qc.add_circuit(residue.dagger(), state_qbits)
        
        compiled_circ = self.backend.get_compiled_circuit(qc)
        psi_T_actual = self.backend.run_circuit(compiled_circ).get_state()[2**self.L:] #The first-half of state vector is zero
        
        np.testing.assert_almost_equal(psi_T_actual, self.psi_T_expected_2nd , decimal=7, err_msg='', verbose=True)

    def G_T_expected(self, trotter_order):
        if(trotter_order == 1):
            return self.G_T_expected_1st
        elif(trotter_order == 2):
            return self.G_T_expected_2nd
        else:
            raise ValueError("Only up to 2nd-order Trotterization is available")

    @parameterized.expand([
        [2, False, False, 1],
        [2, False, True, 1],
        [2, True, False, 1],
        [2, True, True, 1],
        [2, False, True, 2],
        [2, True, True, 2],
        [2, False, True, 3],
        [2, True, True, 3],
        [1, False, False, 1],
        [1, False, True, 1],
        [1, False, False, 2],
        [1, False, True, 2],        
    ])
    def test_hadamard_circuit_real(self, trotter_order, split, optimized, n_controls):
            """Test the real-part Hadamard test circuit for XYZ model."""
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

    @parameterized.expand([
        [2, False, False, 1],
        [2, False, True, 1],
        [2, True, False, 1],
        [2, True, True, 1],
        [2, False, True, 2],
        [2, True, True, 2],
        [2, False, True, 3],
        [2, True, True, 3],
        [1, False, False, 1],
        [1, False, True, 1],
        [1, False, False, 2],
        [1, False, True, 2],        
    ])
    def test_hadamard_circuit_imaginary(self, trotter_order, split, optimized, n_controls):
            """Test the imaginary-part Hadamard test circuit for XYZ model."""
            qc = self.model.make_hadamard_circuit(self.psi_0_circuit, self.dt*self.n_steps, self.n_steps, trotter_order, split, optimized, n_controls, True)
            compiled_circ = self.backend.get_compiled_circuit(qc)
            full_psi = self.backend.run_circuit(compiled_circ).get_state()
            p_ancilla_0 = np.linalg.norm(full_psi[:2**self.L])**2
            p_ancilla_1 = np.linalg.norm(full_psi[2**self.L:])**2
            G_T_actual_imag = p_ancilla_0-p_ancilla_1

            p_zero_state = (abs(full_psi[0])**2+abs(full_psi[-2**self.L])**2)
            G_T_actual_norm2 = 2*p_zero_state-1

            np.testing.assert_almost_equal(G_T_actual_imag,  self.G_T_expected(trotter_order).imag, decimal=7, err_msg=', imaginary parts of G(t) mismatch', verbose=True)
            np.testing.assert_almost_equal(G_T_actual_norm2,  abs(self.G_T_expected(trotter_order))**2, decimal=7, err_msg=', magnitudes of G(t) mismatch', verbose=True)

    @parameterized.expand([
        [1],
        [2],      
    ])
    def test_amplitude_circuit(self, trotter_order):
        """Test the amplitude circuit for XYZ model."""
        qc = self.model.make_amplitude_circuit(self.psi_0_circuit, self.dt*self.n_steps, self.n_steps, trotter_order)
        compiled_circ = self.backend.get_compiled_circuit(qc)
        new_psi = self.backend.run_circuit(compiled_circ).get_state()
        G_T_actual_norm2 = np.linalg.norm(new_psi[0])**2
        np.testing.assert_almost_equal(G_T_actual_norm2,  abs(self.G_T_expected(trotter_order))**2, decimal=7, err_msg=', magnitudes of G(t) mismatch', verbose=True)
        
if __name__ == '__main__':
    unittest.main()
