# from abc import ABC, abstractmethod
import pytket


class FidelityEstimator:
    """Class for estimating the fidelity (aka q-factor) of noisy quantum circuits.

    Parameters
    ----------
    F1: float
        Fidelity of one-qubit gates.
    F2: float
        Fidelity of two-qubit gates.
    """

    def __init__(self, F1: float = 0.99996, F2: float = 0.998):
        self.F1 = F1
        self.F2 = F2

    # @abstractmethod
    # def get_gate_counts(self, compiled_circuit):
    #     raise NotImplementedError()

    def estimate_fidelity(self, qc: pytket.circuit.Circuit) -> float:
        """Estimate the fidelity of a compiled quantum circuit.

        Parameters
        ----------
        qc: :py:class:`pytket.Circuit <pytket._tket.circuit.Circuit>`
            Compiled quantum circuit.

        Returns
        -------
        float
            Fideility (q-factor) of the circuit.
        """

        # one_cout, two_count = self.get_gate_counts(compiled_circuit)
        # return self.F1**one_cout * self.F2**two_count
        return self.F1**qc.n_1qb_gates() * self.F2**qc.n_2qb_gates()


# class HSeriesFidelityEstimator(FidelityEstimator):
#     def __init__(self, F1=0.99996, F2=0.998):
#         super().__init__(F1, F2)

#     def get_gate_counts(self, compiled_circuit):
#         all_count = compiled_circuit.n_gates
#         two_count = compiled_circuit.n_gates_of_type(pytket.circuit.OpType.ZZMax) + compiled_circuit.n_gates_of_type(pytket.circuit.OpType.ZZPhase)
#         one_cout = all_count-two_count
#         return one_cout, two_count
