import stim
from stabstate import *


def remove_errors(protocol: Protocol) -> Protocol:
    return [step for step in protocol if step[0] not in {"RX Error", "RY Error", "RZ Error"}]


def verify_protocol(state: StabState, protocol: Protocol) -> bool:
    circ = protocol_to_stim(protocol)

    circ += stim.Tableau(state.n_qubits).from_stabilizers(state.stab_list).to_circuit().inverse()
    circ.append("M", state.qubit_list)
    sampler = circ.compile_sampler()
    sample = sampler.sample(shots=1000)
    if np.any(np.array(sample)[:, -state.n_qubits:].flatten()):
        # print(np.array2string(np.array(sample)[:, -state.n_qubits:], threshold=np.inf))
        return False
    
    return True
            

def protocol_to_stim(protocol: Protocol) -> stim.Circuit:
    circ_str = ""

    for step in protocol:
            if step[0] in ["RX Error", "RY Error", "RZ Error"]:
                circ_str = circ_str + f"{step[0][1]}_ERROR(0) {step[1]}\n"
            elif step[0] == "MR":
                circ_str = circ_str + f"MR {step[1]}\n"
                for feedforward in step[2]:
                    circ_str = circ_str + f"C{feedforward[0]} rec[-1] {feedforward[1]}\n"
            elif len(step) == 2:
                circ_str = circ_str + f"{step[0]} {step[1]}\n"
            elif len(step) == 3:
                circ_str = circ_str + f"{step[0]} {step[1]} {step[2]}\n"
            else:
                raise Exception(f"Unexpected step: {step}")

    return stim.Circuit(circ_str)


def show_evolution(state: StabState, protocol: Protocol, reversed: bool, back_sub: bool = True, instruction: bool = True, tableau: bool = True, graph: bool = True) -> None:
    state = state.copy()

    if instruction:
        print("INITIAL")
    if tableau:
        state.pprint()
    if graph:
        state.get_graph().draw()

    if reversed:
        protocol = protocol[::-1]

    prev_steps = []
    for step in protocol:
        prev_steps.append(step)
        if step[0] in ["X", "Y", "Z", "H", "S", "S_DAG"]:
            state.clifford(step[0], step[1])
        
        elif step[0] in ["CNOT", "CX", "CZ"]:
            state.clifford(step[0], step[1], step[2])

            state.rref()
            if back_sub:
                state.back_substitution()

            if instruction:
                print(f"DM({step[1]}, {step[2]}): {prev_steps}")
            if tableau:
                state.pprint()
            if graph:
                state.get_graph().draw()

            prev_steps = []

        elif step[0] == "Emission":
            state.clifford("CNOT", step[1], step[2])

            state.rref()
            if back_sub:
                state.back_substitution()

            if instruction:
                print(f"PA({step[1]}, {step[2]}): {prev_steps}")
            if tableau:
                state.pprint()
            if graph:
                state.get_graph().draw()

            prev_steps = []

        elif step[0] == "Measurement":
            state.clifford("H", step[1])
            state.clifford("CNOT", step[1], step[2])

            state.rref()
            if back_sub:
                state.back_substitution()

            if instruction:
                print(f"TRM({step[1]}, {step[2]}): {prev_steps}")
            if tableau:
                state.pprint()
            if graph:
                state.get_graph().draw()

            prev_steps = []

        else:
            raise Exception(f"Unexpected step: {step[0]}")
        
    state.rref()
    if back_sub:
        state.back_substitution()

    if instruction:
        print("FINAL")
    state.pprint()
    state.get_graph().draw()