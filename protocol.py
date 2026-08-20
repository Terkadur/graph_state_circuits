from stabstate import *
from itertools import product, combinations
from time import time_ns


local_gates = ['I', 'H', 'S', 'SH']
local_gate_pairs = list(product(local_gates, repeat=2))

def get_matter_stabs(state: StabState, photon: int = None) -> list[int]:
    allowed_support = state.memory_list + state.emitter_list
    if photon is not None:
        allowed_support.append(photon)

    stabs = []
    for stab in state.qubit_list:
        if set(state.get_support(stab)).issubset(allowed_support):
            stabs.append(stab)

    return stabs

def get_photon_stabs(state: StabState, photon: int) -> list[int]:
    stabs = []
    for stab in state.qubit_list:
        if state.get_support(stab)[0] == photon:
            stabs.append(stab)

    return stabs


def disentangle_stab(state: StabState, stab: int) -> None:
    if len(state.get_support(stab)) != 1:
        raise Exception("Stabilizer doesn't have weight 1")
    
    qubit = state.get_support(stab)[0]
    for other_stab in state.qubit_list:
        if other_stab != stab and qubit in state.get_support(other_stab):
            state.stab_mul(other_stab, stab)


def protocol(state: StabState, recursion_depth: int, emitter_cutoff: int = None, starting_photon: int = None) -> tuple[int, Protocol]:
    state = state.copy()
    if state.n_emitters == 0:
        state.add_qubits(new_emitters=max(state.height_function()))
        

    reversed_protocol = []

    # starting_photon used for future recursions
    if emitter_cutoff is None:
        emitter_cutoff = state.n_qubits
    if starting_photon is None:
        starting_photon = state.n_photons-1
    
    for photon in range(starting_photon, -1, -1):
        # put the state into RREF with partial back substitution
        state.rref()
        back_sub_stabs = get_matter_stabs(state, photon)
        state.back_substitution(stabs = back_sub_stabs)

        # try if photon is free
        new_steps = try_free_photon(state, photon)
        if new_steps is not None:
            reversed_protocol.extend(new_steps)
            continue

        # try if TRM is necessary
        new_steps = try_trm(state, photon, recursion_depth, emitter_cutoff)
        if new_steps is not None:
            reversed_protocol.extend(new_steps)
            continue

        # try if PA is possible without CNOTs
        new_steps = try_pa_wo_cnot(state, photon)
        if new_steps is not None:
            reversed_protocol.extend(new_steps)
            continue

        # try PA with CNOTs
        new_steps = try_pa_w_cnot(state, photon, recursion_depth, emitter_cutoff)
        if new_steps is not None:
            reversed_protocol.extend(new_steps)
            continue

        # TODO: OPTIMIZE BELOW
        raise Exception(f"Absorption of photon {photon} failed")
    
    if state.n_emitters > 0:
        while True:
            new_steps = try_disentangle_emitters(state, recursion_depth, emitter_cutoff)
            if new_steps == []:
                break
            reversed_protocol.extend(new_steps)

    state.rref()

    for stab in state.qubit_list:
        for qubit in state.get_support(stab):
            new_steps = rotate_to_Z(state, stab, qubit)
            reversed_protocol.extend(new_steps)

            if state.stab_list[qubit].sign == -1:
                state.clifford("X", qubit)
                reversed_protocol.append(("X", qubit))

    if state != StabState(n_photons=state.n_photons, n_emitters=state.n_emitters):
        state.pprint()
        raise Exception("Protocol failed")

    return state.n_emitters, reversed_protocol[::-1]


def try_free_photon(state: StabState, photon: int) -> Protocol:
    reversed_steps = []

    # find stabilizer with free photon
    chosen_stab = None
    for stab in get_photon_stabs(state, photon):
        support = state.get_support(stab)
        if len(support) == 1:
            chosen_stab = stab
            break
    if chosen_stab is None:
        return None
    
    # turn photon pauli to Z
    new_steps = rotate_to_Z(state, chosen_stab, photon)
    reversed_steps.extend(new_steps)

    # correct sign of stabilizer
    if state.stab_list[chosen_stab].sign == -1:
        state.clifford("X", photon)
        reversed_steps.append(("X", photon))

    disentangle_stab(state, chosen_stab)

    return reversed_steps


def rotate_to_Z(state: StabState, stab: int, qubit: int) -> Protocol:
    pauli = state.stab_list[stab][qubit]
    if pauli == 0: # identity
        raise Exception("Cannot rotate identity to Z")
    elif pauli == 1: # X
        state.clifford("H", qubit)
        return [("H", qubit)]
    elif pauli == 2: # Y
        state.clifford("S_DAG", qubit)
        state.clifford("H", qubit)
        return [("S", qubit), ("H", qubit)]
    elif pauli == 3: # Z
        return []
    

def try_trm(state: StabState, photon: int, recursion_depth: int, emitter_cutoff: int) -> Protocol:
    reversed_steps = []

    # if no stabs start with photon, trm is needed
    photon_stabs = get_photon_stabs(state, photon)
    if len(photon_stabs) > 0:
        return None
    
    # find stab that acts only on minimal emitters
    emitter_stabs = get_matter_stabs(state)
    chosen_stab = None
    chosen_support = None
    chosen_weight = np.inf
    for stab in emitter_stabs:
        support = state.get_support(stab)
        if len(support) < chosen_weight:
            chosen_stab = stab
            chosen_support = support
            chosen_weight = len(support)
    if chosen_stab is None:
        raise Exception(f"Photon absorption failed on photon {photon}")
    
    # if no stabs act on one emitter, do disentangling routine
    if chosen_weight != 1:
        old_num_subgraphs = state.copy().get_graph_flex()[0].num_subgraphs()
        while True:
            new_num_subgraphs, new_steps = disentangling_subroutine(state, old_num_subgraphs, emitter_cutoff)

            if new_num_subgraphs == old_num_subgraphs:
                break

            reversed_steps.extend(new_steps)
            old_num_subgraphs = new_num_subgraphs

        state.rref()
        emitter_stabs = get_matter_stabs(state)
        state.back_substitution(stabs=emitter_stabs)

        # find stab that acts only on minimal emitters
        emitter_stabs = get_matter_stabs(state)
        chosen_stab = None
        chosen_support = None
        chosen_weight = np.inf
        for stab in emitter_stabs:
            support = state.get_support(stab)
            if len(support) < chosen_weight:
                chosen_stab = stab
                chosen_support = support
                chosen_weight = len(support)
    
    # rotate all paulis to Z
    for chosen_emitter in chosen_support:
        new_steps = rotate_to_Z(state, chosen_stab, chosen_emitter)
        reversed_steps.extend(new_steps)

    # correct sign
    if state.stab_list[chosen_stab].sign == -1:
        state.clifford("X", chosen_support[0])
        reversed_steps.append(("X", chosen_support[0]))

    # if weight is 1, just do trm
    if chosen_weight == 1:
        chosen_emitter = chosen_support[0]

        # perform time reversed measurement
        state.clifford("H", chosen_emitter)
        state.clifford("CNOT", chosen_emitter, photon)
        reversed_steps.append(("MR", chosen_emitter, [("X", photon)]))

        # rotate emitter and photon to Z
        state.clifford("H", chosen_emitter)
        state.clifford("H", photon)
        reversed_steps.extend([("H", chosen_emitter), ("H", photon)])

        if state.stab_list[chosen_stab].sign == -1:
            state.clifford("X", photon)
            reversed_steps.append(("X", photon))

        state.clifford("CNOT", chosen_emitter, photon)
        reversed_steps.append(("CNOT", chosen_emitter, photon))
        
        disentangle_stab(state, chosen_stab)

        return reversed_steps
    
    # otherwise, do lookahead to check for optimal emitter
    search_emitters = chosen_support[:emitter_cutoff]
    if recursion_depth == 0:
        chosen_state = None
        chosen_emitter = None
        chosen_steps = None
        chosen_edge_num = np.inf
        for index, emitter in enumerate(search_emitters):
            new_steps = []
            state_copy = state.copy()
            other_emitters = chosen_support[:index] + chosen_support[index+1:]
            for other_emitter in other_emitters:
                state_copy.clifford("CNOT", other_emitter, emitter)
                new_steps.append(("CNOT", other_emitter, emitter))

            # perform time reversed measurement
            state_copy.clifford("H", emitter)
            state_copy.clifford("CNOT", emitter, photon)
            new_steps.append(("MR", emitter, [("X", photon)]))

            # rotate emitter and photon to Z
            state_copy.clifford("H", emitter)
            state_copy.clifford("H", photon)
            new_steps.extend([("H", emitter), ("H", photon)])

            if state_copy.stab_list[chosen_stab].sign == -1:
                state_copy.clifford("X", photon)
                new_steps.append(("X", photon))

            state_copy.clifford("CNOT", emitter, photon)
            new_steps.append(("CNOT", emitter, photon))

            disentangle_stab(state_copy, chosen_stab)

            edge_num = len(state_copy.copy().get_graph_flex()[0].edges)
            if edge_num < chosen_edge_num:
                chosen_state = state_copy
                chosen_emitter = emitter
                chosen_steps = new_steps
                chosen_edge_num = edge_num

        state.stab_list = chosen_state.stab_list
        reversed_steps.extend(chosen_steps)

        return reversed_steps
    
    else:
        chosen_state = None
        chosen_emitter = None
        chosen_steps = None
        chosen_cnot_num = np.inf
        for index, emitter in enumerate(search_emitters):
            new_steps = []
            state_copy = state.copy()
            other_emitters = chosen_support[:index] + chosen_support[index+1:]
            for other_emitter in other_emitters:
                state_copy.clifford("CNOT", other_emitter, emitter)
                new_steps.append(("CNOT", other_emitter, emitter))

            # perform time reversed measurement
            state_copy.clifford("H", emitter)
            state_copy.clifford("CNOT", emitter, photon)
            new_steps.append(("MR", emitter, [("X", photon)]))

            # rotate emitter and photon to Z
            state_copy.clifford("H", emitter)
            state_copy.clifford("H", photon)
            new_steps.extend([("H", emitter), ("H", photon)])

            if state_copy.stab_list[chosen_stab].sign == -1:
                state_copy.clifford("X", photon)
                new_steps.append(("X", photon))

            state_copy.clifford("CNOT", emitter, photon)
            new_steps.append(("CNOT", emitter, photon))

            disentangle_stab(state_copy, chosen_stab)

            _, future_protocol = protocol(state_copy, recursion_depth=recursion_depth-1, emitter_cutoff=emitter_cutoff, starting_photon=photon-1)
            cnot_num = 0
            for step in future_protocol:
                if step[0] in {"CX", "CNOT", "CY", "CZ", "CPHASE"}:
                    cnot_num += 1

            if cnot_num < chosen_cnot_num:
                chosen_state = state_copy
                chosen_emitter = emitter
                chosen_steps = new_steps
                chosen_cnot_num = cnot_num

        state.stab_list = chosen_state.stab_list
        reversed_steps.extend(chosen_steps)

        return reversed_steps


def try_pa_wo_cnot(state: StabState, photon: int) -> Protocol:
    reversed_steps = []

    # find stab with only photon and an emitter
    photon_stabs = get_photon_stabs(state, photon)
    chosen_stab = None
    chosen_emitter = None
    for stab in photon_stabs:
        support = state.get_support(stab)
        if support[1] in state.emitter_list and len(support) == 2:
            chosen_stab = stab
            chosen_emitter = support[1]
    if chosen_stab is None:
        return None
    
    # turn all paulis to Z
    new_steps = rotate_to_Z(state, chosen_stab, chosen_emitter)
    reversed_steps.extend(new_steps)
    new_steps = rotate_to_Z(state, chosen_stab, photon)
    reversed_steps.extend(new_steps)

    if state.stab_list[chosen_stab].sign == -1:
        state.clifford("X", photon)
        reversed_steps.append(("X", photon))

    state.clifford("CNOT", chosen_emitter, photon)
    reversed_steps.append(("CNOT", chosen_emitter, photon))

    disentangle_stab(state, chosen_stab)

    return reversed_steps


def try_pa_w_cnot(state: StabState, photon: int, recursion_depth: int, emitter_cutoff: int) -> Protocol:
    reversed_steps = []
    
    old_num_subgraphs = state.copy().get_graph_flex()[0].num_subgraphs()

    while True:
        new_num_subgraphs, new_steps = disentangling_subroutine(state, old_num_subgraphs, emitter_cutoff, photon=photon)

        if new_num_subgraphs == old_num_subgraphs:
            break

        reversed_steps.extend(new_steps)
        old_num_subgraphs = new_num_subgraphs

    state.rref()
    photon_stabs = get_photon_stabs(state, photon)
    emitter_stabs = get_matter_stabs(state)
    state.back_substitution(stabs=photon_stabs + emitter_stabs)

    # try PA without CNOTs
    new_steps = try_pa_wo_cnot(state, photon)
    if new_steps is not None:
        reversed_steps.extend(new_steps)
        return reversed_steps
    
    # find photon stabilizer with minimal weight
    photon_stabs = get_photon_stabs(state, photon)
    chosen_stab = None
    chosen_emitters = None
    chosen_weight = np.inf
    for stab in photon_stabs:
        support = state.get_support(stab)
        if len(support) < chosen_weight:
            chosen_stab = stab
            chosen_emitters = support[1:]
            chosen_weight = len(support)

    # rotate all paulis to Z
    new_steps = rotate_to_Z(state, chosen_stab, photon)
    reversed_steps.extend(new_steps)
    for chosen_emitter in chosen_emitters:
        new_steps = rotate_to_Z(state, chosen_stab, chosen_emitter)
        reversed_steps.extend(new_steps)

    if state.stab_list[chosen_stab].sign == -1:
        state.clifford("X", photon)
        reversed_steps.append(("X", photon))

    # check for optimal emitter
    search_emitters = chosen_emitters[:emitter_cutoff]
    if recursion_depth == 0:
        chosen_state = None
        chosen_emitter = None
        chosen_steps = None
        chosen_edge_num = np.inf
        for index, emitter in enumerate(search_emitters):
            new_steps = []
            state_copy = state.copy()
            other_emitters = chosen_emitters[:index] + chosen_emitters[index+1:]
            for other_emitter in other_emitters:
                state_copy.clifford("CNOT", other_emitter, emitter)
                new_steps.append(("CNOT", other_emitter, emitter))

            state_copy.clifford("CNOT", emitter, photon)
            new_steps.append(("CNOT", emitter, photon))

            disentangle_stab(state_copy, chosen_stab)

            edge_num = len(state_copy.copy().get_graph_flex()[0].edges)
            if edge_num < chosen_edge_num:
                chosen_state = state_copy
                chosen_emitter = emitter
                chosen_steps = new_steps
                chosen_edge_num = edge_num

        state.stab_list = chosen_state.stab_list
        reversed_steps.extend(chosen_steps)

        return reversed_steps

    else:
        chosen_state = None
        chosen_emitter = None
        chosen_steps = None
        chosen_cnot_num = np.inf
        for index, emitter in enumerate(search_emitters):
            new_steps = []
            state_copy = state.copy()
            other_emitters = chosen_emitters[:index] + chosen_emitters[index+1:]
            for other_emitter in other_emitters:
                state_copy.clifford("CNOT", other_emitter, emitter)
                new_steps.append(("CNOT", other_emitter, emitter))

            state_copy.clifford("CNOT", emitter, photon)
            new_steps.append(("CNOT", emitter, photon))

            disentangle_stab(state_copy, chosen_stab)
            
            _, future_protocol = protocol(state_copy, recursion_depth=recursion_depth-1, emitter_cutoff=emitter_cutoff, starting_photon=photon-1)
            cnot_num = 0
            for step in future_protocol:
                if step[0] in {"CX", "CNOT", "CY", "CZ", "CPHASE"}:
                    cnot_num += 1
                
            if cnot_num < chosen_cnot_num:
                chosen_state = state_copy
                chosen_emitter = emitter
                chosen_steps = new_steps
                chosen_cnot_num = cnot_num

        state.stab_list = chosen_state.stab_list
        reversed_steps.extend(chosen_steps)

        return reversed_steps


    


def disentangling_subroutine(state: StabState, old_num_subgraphs: int, emitter_cutoff: int,  photon: int = None) -> tuple[int, Protocol]:
    reversed_steps = []

    state.rref()
    emitter_stabs = get_matter_stabs(state)
    state.back_substitution(stabs=emitter_stabs)

    # check if there is a two-emitter stabilizer to disentangle
    chosen_stab = None
    chosen_support = None
    for stab in emitter_stabs:
        support = state.get_support(stab)
        if len(support) == 2:
            chosen_stab = stab
            chosen_support = support
            break
    if chosen_stab is not None:
        emitter1 = chosen_support[0]
        emitter2 = chosen_support[1]
        pauli1 = state.get_pauli(chosen_stab, emitter1)
        pauli2 = state.get_pauli(chosen_stab, emitter2)
        new_steps = disentangle_two_emitters(state, emitter1, emitter2, pauli1, pauli2)
        return old_num_subgraphs+1, new_steps

    # if photon is None find emitter stabilizer with minimal emitters
    # if photon isn't None find photon stabilizer with minimal emitters
    search_stabs = None
    if photon is None:
        search_stabs = emitter_stabs
        state.back_substitution(stabs=emitter_stabs)
    else:
        photon_stabs = get_photon_stabs(state, photon)
        search_stabs = photon_stabs
        state.back_substitution(stabs=(emitter_stabs + photon_stabs))
    chosen_emitters = None
    chosen_weight = np.inf
    for stab in search_stabs:
        support = state.get_support(stab)
        if len(support) < chosen_weight:
            chosen_emitters = support[1:]
            chosen_weight = len(support)
    chosen_emitters = chosen_emitters[:emitter_cutoff]

    num_loop = 0
    for emitter1, emitter2 in combinations(chosen_emitters, 2):
        for gate1, gate2 in local_gate_pairs:

            num_loop += 1
            state_copy = state.copy()
            
            if gate1 == "H":
                state_copy.clifford("H", emitter1)
            elif gate1 == "S":
                state_copy.clifford("S_DAG", emitter1)
            elif gate1 == "SH":
                state_copy.clifford("S_DAG", emitter1)
                state_copy.clifford("H", emitter1)
            
            if gate2 == "H":
                state_copy.clifford("H", emitter2)
            elif gate2 == "S":
                state_copy.clifford("S_DAG", emitter2)
            elif gate2 == "SH":
                state_copy.clifford("S_DAG", emitter2)
                state_copy.clifford("H", emitter2)

            state_copy.clifford("CNOT", emitter1, emitter2)

            new_num_subgraphs = state_copy.copy().get_graph_flex()[0].num_subgraphs()
            if new_num_subgraphs > old_num_subgraphs:
                state.stab_list = state_copy.stab_list

                if gate1 == "H":
                    reversed_steps.append(("H", emitter1))
                elif gate1 == "S":
                    reversed_steps.append(("S", emitter1))
                elif gate1 == "SH":
                    reversed_steps.append(("S", emitter1))
                    reversed_steps.append(("H", emitter1))

                if gate2 == "H":
                    reversed_steps.append(("H", emitter2))
                elif gate2 == "S":
                    reversed_steps.append(("S", emitter2))
                elif gate2 == "SH":
                    reversed_steps.append(("S", emitter2))
                    reversed_steps.append(("H", emitter2))

                reversed_steps.append(("CNOT", emitter1, emitter2))
                return new_num_subgraphs, reversed_steps
            
            # try reversing cnot control and target
            state_copy = state.copy()
            
            if gate1 == "H":
                state_copy.clifford("H", emitter1)
            elif gate1 == "S":
                state_copy.clifford("S_DAG", emitter1)
            elif gate1 == "SH":
                state_copy.clifford("S_DAG", emitter1)
                state_copy.clifford("H", emitter1)
            
            if gate2 == "H":
                state_copy.clifford("H", emitter2)
            elif gate2 == "S":
                state_copy.clifford("S_DAG", emitter2)
            elif gate2 == "SH":
                state_copy.clifford("S_DAG", emitter2)
                state_copy.clifford("H", emitter2)

            state_copy.clifford("CNOT", emitter2, emitter1)

            new_num_subgraphs = state_copy.copy().get_graph_flex()[0].num_subgraphs()
            if new_num_subgraphs > old_num_subgraphs:
                state.stab_list = state_copy.stab_list

                if gate1 == "H":
                    reversed_steps.append(("H", emitter1))
                elif gate1 == "S":
                    reversed_steps.append(("S", emitter1))
                elif gate1 == "SH":
                    reversed_steps.append(("S", emitter1))
                    reversed_steps.append(("H", emitter1))

                if gate2 == "H":
                    reversed_steps.append(("H", emitter2))
                elif gate2 == "S":
                    reversed_steps.append(("S", emitter2))
                elif gate2 == "SH":
                    reversed_steps.append(("S", emitter2))
                    reversed_steps.append(("H", emitter2))

                reversed_steps.append(("CNOT", emitter2, emitter1))
                return new_num_subgraphs, reversed_steps
            
    return old_num_subgraphs, reversed_steps

    

def disentangle_two_emitters(state: StabState, emitter1: int, emitter2: int, pauli1: str, pauli2: str) -> Protocol:
    reversed_steps = []

    if pauli1 == "X" and pauli2 == "X":
        state.clifford("CNOT", emitter1, emitter2)
        reversed_steps.append(("CNOT", emitter1, emitter2))
    elif pauli1 == "X" and pauli2 == "Y":
        state.clifford("CNOT", emitter2, emitter1)
        reversed_steps.append(("CNOT", emitter2, emitter1))
    elif pauli1 == "X" and pauli2 == "Z":
        state.clifford("CZ", emitter1, emitter2)
        reversed_steps.extend([("CZ", emitter1, emitter2)])
    elif pauli1 == "Y" and pauli2 == "X":
        state.clifford("CNOT", emitter1, emitter2)
        reversed_steps.append(("CNOT", emitter1, emitter2))
    elif pauli1 == "Y" and pauli2 == "Y":
        state.clifford("S_DAG", emitter1)
        state.clifford("CNOT", emitter2, emitter1)
        reversed_steps.extend([("S", emitter1), ("CNOT", emitter2, emitter1)])
    elif pauli1 == "Y" and pauli2 == "Z":
        state.clifford("CNOT", emitter2, emitter1)
        reversed_steps.append(("CNOT", emitter2, emitter1))
    elif pauli1 == "Z" and pauli2 == "X":
        state.clifford("CZ", emitter1, emitter2)
        reversed_steps.append(("CZ", emitter1, emitter2))
    elif pauli1 == "Z" and pauli2 == "Y":
        state.clifford("CNOT", emitter1, emitter2)
        reversed_steps.append(("CNOT", emitter1, emitter2))
    elif pauli1 == "Z" and pauli2 == "Z":
        state.clifford("CNOT", emitter1, emitter2)
        reversed_steps.append(("CNOT", emitter1, emitter2))

    return reversed_steps


def try_disentangle_two_emitters(state: StabState) -> Protocol: # TODO optimize?
    reversed_steps = []

    state.rref()
    emitter_stabs = get_matter_stabs(state)
    state.back_substitution(stabs=emitter_stabs)

    # look for stabilizer with minimal emitters (at least two)
    chosen_stab = None
    chosen_support = None
    chosen_weight = np.inf
    for stab in emitter_stabs:
        support = state.get_support(stab)
        if len(support) < chosen_weight and len(support) > 1:
            chosen_stab = stab
            chosen_support = support
            chosen_weight = len(support)
    if chosen_stab is None:
        return None

    X_emitters = []
    Y_emitters = []
    Z_emitters = []
    for emitter in chosen_support:
        if state.get_pauli(chosen_stab, emitter) == "X":
            X_emitters.append(emitter)
        elif state.get_pauli(chosen_stab, emitter) == "Y":
            Y_emitters.append(emitter)
        elif state.get_pauli(chosen_stab, emitter) == "Z":
            Z_emitters.append(emitter)
    
    if len(Z_emitters) == 2:
        state.clifford("CNOT", Z_emitters[0], Z_emitters[1])
        reversed_steps.append(("CNOT", Z_emitters[0], Z_emitters[1]))
        return reversed_steps

    for emitter in X_emitters:
        state.clifford("H", emitter)
        reversed_steps.append(("H", emitter))     
        Z_emitters.append(emitter)
        if len(Z_emitters) == 2:
            break
    if len(Z_emitters) == 2:
        state.clifford("CNOT", Z_emitters[0], Z_emitters[1])
        reversed_steps.append(("CNOT", Z_emitters[0], Z_emitters[1]))
        return reversed_steps
    
    for emitter in Y_emitters:
        state.clifford("S_DAG", emitter)
        state.clifford("H", emitter)
        reversed_steps.append(("S", emitter))
        reversed_steps.append(("H", emitter))
        
        Z_emitters.append(emitter)
        if len(Z_emitters) == 2:
            break
    
    state.clifford("CNOT", Z_emitters[0], Z_emitters[1])
    reversed_steps.append(("CNOT", Z_emitters[0], Z_emitters[1]))

    return reversed_steps


# TODO: use disentangle_two_emitters function
def try_disentangle_emitters(state: StabState, recursion_depth: int, emitter_cutoff: int) -> Protocol:
    reversed_steps = []

    state.rref()
    emitter_stabs = get_matter_stabs(state)
    state.back_substitution(stabs=emitter_stabs)
    
    # if no stabs act on one emitter, do disentangling routine
    old_num_subgraphs = state.copy().get_graph_flex()[0].num_subgraphs()
    while True:
        new_num_subgraphs, new_steps = disentangling_subroutine(state, old_num_subgraphs, emitter_cutoff)

        if new_num_subgraphs == old_num_subgraphs:
            break

        reversed_steps.extend(new_steps)
        old_num_subgraphs = new_num_subgraphs

    state.rref()
    emitter_stabs = get_matter_stabs(state)
    state.back_substitution(stabs=emitter_stabs)

    # find stab that acts only on minimal emitters but greater than 1
    emitter_stabs = get_matter_stabs(state)
    chosen_stab = None
    chosen_support = None
    chosen_weight = np.inf
    for stab in emitter_stabs:
        support = state.get_support(stab)
        if len(support) < chosen_weight and len(support) > 1:
            chosen_stab = stab
            chosen_support = support
            chosen_weight = len(support)
    if chosen_stab is None:
        return reversed_steps
    
    # rotate all paulis to Z
    for chosen_emitter in chosen_support:
        new_steps = rotate_to_Z(state, chosen_stab, chosen_emitter)
        reversed_steps.extend(new_steps)

    # correct sign
    if state.stab_list[chosen_stab].sign == -1:
        state.clifford("X", chosen_support[0])
        reversed_steps.append(("X", chosen_support[0]))

    
    # do lookahead to check for optimal emitter
    search_emitters = chosen_support[:emitter_cutoff]
    if recursion_depth == 0:
        chosen_state = None
        chosen_emitter = None
        chosen_steps = None
        chosen_edge_num = np.inf
        for index, emitter in enumerate(search_emitters):
            new_steps = []
            state_copy = state.copy()
            other_emitters = chosen_support[:index] + chosen_support[index+1:]
            for other_emitter in other_emitters:
                state_copy.clifford("CNOT", other_emitter, emitter) 
                new_steps.append(("CNOT", other_emitter, emitter))

            disentangle_stab(state_copy, chosen_stab)

            edge_num = len(state_copy.copy().get_graph_flex()[0].edges)
            if edge_num < chosen_edge_num:
                chosen_state = state_copy
                chosen_emitter = emitter
                chosen_steps = new_steps
                chosen_edge_num = edge_num

        state.stab_list = chosen_state.stab_list
        reversed_steps.extend(chosen_steps)

        return reversed_steps
    
    else:
        chosen_state = None
        chosen_emitter = None
        chosen_steps = None
        chosen_cnot_num = np.inf
        for index, emitter in enumerate(search_emitters):
            new_steps = []
            state_copy = state.copy()
            other_emitters = chosen_support[:index] + chosen_support[index+1:]
            for other_emitter in other_emitters:
                state_copy.clifford("CNOT", other_emitter, emitter)
                new_steps.append(("CNOT", other_emitter, emitter))

            disentangle_stab(state_copy, chosen_stab)

            _, future_protocol = protocol(state_copy, recursion_depth=recursion_depth-1, emitter_cutoff=emitter_cutoff, starting_photon=0)
            cnot_num = 0
            for step in future_protocol:
                if step[0] in {"CX", "CNOT", "CY", "CZ", "CPHASE"}:
                    cnot_num += 1

            if cnot_num < chosen_cnot_num:
                chosen_state = state_copy
                chosen_emitter = emitter
                chosen_steps = new_steps
                chosen_cnot_num = cnot_num

        state.stab_list = chosen_state.stab_list
        reversed_steps.extend(chosen_steps)

        return reversed_steps