import numpy as np
from pytket.circuit import Circuit, CircBox, QControlBox, Unitary2qBox, Unitary1qBox

def append_CRz(qc, angle, cnt_qbit, trgt_qbit):
    """Appends a controlled-CRz gate with a symbolic angle (specified in half-turns) on the wires for the specified qubits."""
    qc.ZZPhase(-angle/2, cnt_qbit, trgt_qbit)
    qc.Rz(angle/2, trgt_qbit)

def append_CRx(qc, angle, cnt_qbit, trgt_qbit):
    """Appends a controlled-CRx gate with a symbolic angle (specified in half-turns) on the wires for the specified qubits."""
    qc.H(trgt_qbit)
    append_CRz(qc, angle, cnt_qbit, trgt_qbit)
    qc.H(trgt_qbit)

def append_CRy(qc, angle, cnt_qbit, trgt_qbit):
    """Appends a controlled-CRy gate with a symbolic angle (specified in half-turns) on the wires for the specified qubits."""
    qc.Sdg(trgt_qbit)
    qc.H(trgt_qbit)
    append_CRz(qc, angle, cnt_qbit, trgt_qbit)
    qc.H(trgt_qbit)
    qc.S(trgt_qbit)

def append_CZZPhase(qc, angle, cnt_qbit, trgt_qbit1, trgt_qbit2):
    """Appends a controlled-ZZ gate with a symbolic angle (specified in half-turns) on the wires for the specified qubits."""
    qc.CX(trgt_qbit1, trgt_qbit2)
    append_CRz(qc, angle, cnt_qbit, trgt_qbit2)
    qc.CX(trgt_qbit1, trgt_qbit2)

def append_CXXPhase(qc, angle, cnt_qbit, trgt_qbit1, trgt_qbit2):
    """Appends a controlled-XX gate with a symbolic angle (specified in half-turns) on the wires for the specified qubits."""
    qc.H(trgt_qbit1)
    qc.H(trgt_qbit2)
    append_CZZPhase(qc, angle, cnt_qbit, trgt_qbit1, trgt_qbit2)
    qc.H(trgt_qbit1)
    qc.H(trgt_qbit2)

def append_CYYPhase(qc, angle, cnt_qbit, trgt_qbit1, trgt_qbit2):
    """Appends a controlled-YY gate with a symbolic angle (specified in half-turns) on the wires for the specified qubits."""
    qc.Sdg(trgt_qbit1)
    qc.Sdg(trgt_qbit2)
    qc.H(trgt_qbit1)
    qc.H(trgt_qbit2)
    append_CZZPhase(qc, angle, cnt_qbit, trgt_qbit1, trgt_qbit2)
    qc.H(trgt_qbit1)
    qc.H(trgt_qbit2)
    qc.S(trgt_qbit1)
    qc.S(trgt_qbit2)

def append_XXPhaseYYPhase(qc, angle, qbit1, qbit2):
    """Appends a e^(-i angle * (XX+YY)/2) gate with a symbolic angles (specified in half-turns) on the wires for the specified qubits."""
    """Equivalent to XXPhase(angle)YYPhase(angle)"""
    angle2 = -angle*np.pi
    U = np.array([[1, 0                , 0                , 0],
                  [0, np.cos(angle2)   , 1j*np.sin(angle2), 0],
                  [0, 1j*np.sin(angle2), np.cos(angle2)   , 0],
                  [0, 0                , 0                , 1]])

    u2box = Unitary2qBox(U)                
    qc.add_unitary2qbox(u2box, qbit1, qbit2)

def append_CXXPhaseYYPhase(qc,alpha,beta,cnt_qbit,trgt_qbit1, trgt_qbit2):
    """Appends a controlled exp[-i( alpha XX +beta YY)/2] gate with a symbolic angles (specified in half-turns) on the wires for the specified qubits."""
    """Equivalent to Controlled XXPhase(alpha)YYPhase(beta)"""
    qc.Rx(1/2, trgt_qbit1)
    qc.Rx(1/2, trgt_qbit2)
    qc.CX(trgt_qbit1, trgt_qbit2)
    append_CRx(qc, alpha, cnt_qbit, trgt_qbit1)
    append_CRz(qc, beta, cnt_qbit, trgt_qbit2)
    qc.CX(trgt_qbit1, trgt_qbit2)
    qc.Rx(-1/2, trgt_qbit1)
    qc.Rx(-1/2, trgt_qbit2)


def append_CXXPhaseYYPhaseZZPhase(qc, alpha, beta, gamma, cnt_qbit, trgt_qbit1, trgt_qbit2):
    """Appends a controlled exp[-i( alpha XX +beta YY + gamma ZZ)/2] gate with a symbolic angles (specified in half-turns) on the wires for the specified qubits."""
    """Equivalent to Controlled XXPhase(alpha)YYPhase(beta)ZZPhase(gamma) """
    qc.CX(trgt_qbit1, trgt_qbit2)
    append_CRx(qc, alpha, cnt_qbit, trgt_qbit1)
    append_CRz(qc, gamma, cnt_qbit, trgt_qbit2)
    qc.H(trgt_qbit1)
    qc.CX(trgt_qbit1,trgt_qbit2)
    qc.S(trgt_qbit1)
    append_CRz(qc, -beta, cnt_qbit, trgt_qbit2)
    qc.H(trgt_qbit1)
    qc.CX(trgt_qbit1, trgt_qbit2)
    qc.Rx(-1/2, trgt_qbit1)
    qc.Rx(1/2, trgt_qbit2)

def make_hadamard_circuit(psi_circuit, U_circuit, n_controls_U, imaginary, phase_shift = 0):
    """Return a tket circuit that does the hadamard test for <psi| e^{i phase_shift}  U | psi>
    
    Parameters
    ----------
    psi_circuit: tket circuit
        Circuit for transforming the |0...0> state into |psi>

    U_circuit: tket circuit
        Circuit for the unitary operator U (possibly controlled).

    n_controls_U: int
        The number of control qbits in U_circuit. The control bits are assumed to be the first ones in the circuit.
        If the number is zero, U_circuit will be automatically controlled using a single ancilla qbit with a QControlBox.
        Providing an optimized controlled-U can be more efficent than treating U as a blackbox.

    imaginary: bool
        If 'True', return the circuit measuring the imaginary part of <psi| U | psi>. Otherwise, return the circuit for the real part.

    phase_shift: float
        Value to be used for shifting the phase of overlap.
    """
    n_controls = n_controls_U if n_controls_U > 0 else 1
    qc = Circuit(n_controls+psi_circuit.n_qubits)
    controls = list(range(0, n_controls))
    psi = list(range(n_controls, n_controls+psi_circuit.n_qubits))
    qc.H(controls[0])
    if(imaginary):
        qc.Sdg(controls[0])
    for other_cnt in controls[1:]:
        qc.CX(controls[0], other_cnt)
    qc.add_circuit(psi_circuit, psi)
    if(n_controls_U > 0):
        qc.add_circuit(U_circuit, controls+psi)
    else:
        qc.add_qcontrolbox(QControlBox(CircBox(U_circuit), 1), controls+psi)
    for other_cnt in controls[1:]:
        qc.CX(controls[0], other_cnt)
    if(phase_shift!= 0):
        qc.Rz(phase_shift/np.pi, controls[0])
    qc.H(controls[0])
    for other_cnt in controls[1:]:
        qc.CX(controls[0], other_cnt)
    qc.add_circuit(psi_circuit.dagger(), psi)
    return qc


def log_cat_preperation_circuit(theta,psi_bit_string):
    L=len(psi_bit_string)

    H_theta=(1.0/np.sqrt(2.0))*np.array([[1,np.exp(-1j*theta)],[np.exp(1j*theta),-1]])
    u1box=Unitary1qBox(H_theta)

    circ=Circuit(L)
    control_list=[]
    for i,s in enumerate(psi_bit_string):
        if s==1:
            control_list.append(i)
    if len(control_list)>0:
        circ.add_unitary1qbox(u1box,control_list[0])#First Hadamard rotation
    pairs=[]
    if len(control_list)>1:
        flipped=[control_list[0]]
        to_flip=control_list[1:]
        while to_flip!=[]:
            nf=np.min([len(flipped),len(to_flip)])
            for jq in range(nf): 
                pairs.append((flipped[jq],to_flip[jq]))
            for jq in range(nf):
                flipped.append(to_flip[jq])
            for jq in range(nf):
                to_flip.pop(0)

    for a_pair in pairs: 
        circ.CX(control_qubit=a_pair[0],target_qubit=a_pair[1])
    return circ

def linear_cat_preperation_circuit(theta,psi_bit_string):
    L=len(psi_bit_string)

    H_theta=(1.0/np.sqrt(2.0))*np.array([[1,np.exp(-1j*theta)],[np.exp(1j*theta),-1]])
    u1box=Unitary1qBox(H_theta)

    circ=Circuit(L)

    control_qubit=-2
    for i,s in enumerate(psi_bit_string):
        if s==1:
            if control_qubit>-1:
                circ.CX(control_qubit=control_qubit,target_qubit=i)
            else:
                circ.add_unitary1qbox(u1box,i)
                control_qubit=i
    return circ

def make_cat_circuit(U_circuit, psi_bitstring, theta, phase_shift = 0, log_preperation = True):
    """Return a tket circuit that does the cat state circuit V^†(0) U V(theta+phase_shift) |0..0>
    where V(phi) |0..0> = (|0..0>+e^(i phi) |psi>)/sqrt(2)
    
    The probability of measuring |0..0> in this circuit is: 1/4 * |<0..0| U |0..0>+e^(i theta) <psi| U e^(i phase_shift) |psi>|^2

    Parameters
    ----------
    U_circuit: tket circuit
        Circuit for the unitary operator U.

    psi_bitstring: List of int
        Bit string representation of the product state |psi>

    theta: float
        Value of pahse difference between |0..0> and |psi> in the cat state.

    phase_shift: float
        Value to be used for shifting the phase of overlap.

    log_preperation: bool, optional
        Whether to use a circuit of logarithmic depth for prepering the cat state. Deafult value is true (recommended)
    """
    if(log_preperation):
        cat_prep_circuit1 = log_cat_preperation_circuit(theta + phase_shift, psi_bitstring)
        cat_prep_circuit2 = log_cat_preperation_circuit(0, psi_bitstring)
    else:
        cat_prep_circuit1 = linear_cat_preperation_circuit(theta + phase_shift, psi_bitstring)
        cat_prep_circuit2 = linear_cat_preperation_circuit(0, psi_bitstring)


    qbits_count = len(psi_bitstring)
    all_qbits = list(range(qbits_count))

    qc = Circuit(qbits_count)
    qc.add_circuit(cat_prep_circuit1, all_qbits)
    qc.add_circuit(U_circuit, all_qbits)
    qc.add_circuit(cat_prep_circuit2.dagger(), all_qbits)

    return qc