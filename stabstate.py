from graph import *
import numpy as np
from stim import PauliString, Tableau
from galois import GF2

type Protocol = list[tuple[str, int] | tuple[str, int, int]]


def pauli_to_vector(pauli_string: PauliString) -> np.array:
    n_photons = len(pauli_string)
    vector = np.zeros(2*n_photons, dtype=int)

    for index, pauli in enumerate(pauli_string):
        if pauli == 0:
            pass
        elif pauli == 1:
            vector[index] = 1
        elif pauli == 2:
            vector[index] = 1
            vector[index + n_photons] = 1
        elif pauli == 3:
            vector[index + n_photons] = 1

    return vector


class StabState:

    def __init__(self, graph: Graph = None, n_photons: int = 0, n_memories: int = 0, n_emitters: int = 0, stabs: list[str] = None):
        if graph is not None:
            self.n_photons = graph.n_photons
            self.n_memories = graph.n_memories
            self.n_emitters = graph.n_emitters

            self.stab_list = [PauliString("I"*self.n_qubits) for _ in self.qubit_list]
            for qubit, stab in enumerate(self.stab_list):
                if len(graph.get_nbhd(qubit)) == 0:
                    stab[qubit] = "Z"
                else:
                    stab[qubit] = "X"
            for edge in graph.edges:
                qubit1 = edge[0]
                qubit2 = edge[1]
                self.stab_list[qubit1][qubit2] = "Z"
                self.stab_list[qubit2][qubit1] = "Z"

        elif stabs is not None:
            self.n_photons = n_photons
            self.n_memories = n_memories
            self.n_emitters = n_emitters


            if len(stabs) != self.n_qubits:
                raise Exception(f"Expected {self.n_qubits} stabilizers but got {len(stabs)}")
            for stab in stabs:
                if len(PauliString(stab)) != self.n_qubits:
                    raise Exception(f"Stabilizer {stab} is not length {self.n_qubits}")

            self.stab_list = [PauliString(stab) for stab in stabs]
            try:
                stim.Tableau.from_stabilizers(self.stab_list)
            except:
                raise Exception("Stabilizers do not form a valid stabilizer state")
            

        else:
            self.n_photons = n_photons
            self.n_memories = n_memories
            self.n_emitters = n_emitters
            
            self.stab_list = [PauliString("I"*self.n_qubits) for _ in self.qubit_list]
            for qubit, stab in enumerate(self.stab_list):
                stab[qubit] = "Z"

    @property
    def n_qubits(self):
        return self.n_photons + self.n_memories + self.n_emitters
    
    @property
    def photon_list(self):
        return list(range(self.n_photons))
    
    @property
    def memory_list(self):
        return list(range(self.n_photons, self.n_photons + self.n_memories))
    
    @property
    def emitter_list(self):
        return list(range(self.n_photons + self.n_memories, self.n_photons + self.n_memories + self.n_emitters))
    
    @property
    def qubit_list(self):
        return list(range(self.n_photons + self.n_memories + self.n_emitters))

    def __repr__(self) -> str:
        return str(self.stab_list)

    def __eq__(self, other) -> bool:
        if self.n_photons != other.n_photons:
            return False
        
        if self.n_memories != other.n_memories:
            return False
        
        if self.n_emitters != other.n_emitters:
            return False
        

        tab1 = stim.Tableau.from_stabilizers(self.stab_list)
        tab1 = tab1.inverse()
        for gen in other.stab_list:
            new_gen = tab1(gen)
            if new_gen.sign == -1:
                return False
            if 1 in new_gen or 2 in new_gen:
                return False
        
        return True

    def copy(self):
        state = StabState()

        state.n_photons = self.n_photons
        state.n_memories = self.n_memories
        state.n_emitters = self.n_emitters

        state.stab_list = [stab.copy() for stab in self.stab_list]

        return state
    
    def randomize(self):
        clif = stim.Tableau.random(self.n_qubits)
    
        self.stab_list = []
        for i in range(self.n_qubits):
            self.stab_list.append(clif.z_output(i))
    
    def add_qubits(self, new_photons: int = 0, new_memories: int = 0, new_emitters: int = 0):
        # add spaces to old stabilizers
        for stab_index, stab in enumerate(self.stab_list):
            new_stab = stab[: self.n_photons].copy()
            new_stab += PauliString("I"*new_photons)
            new_stab += stab[self.n_photons : self.n_photons + self.n_memories]
            new_stab += PauliString("I"*new_memories)
            new_stab += stab[self.n_photons + self.n_memories :]
            new_stab += PauliString("I"*new_emitters)
            new_stab *= stab.sign
            self.stab_list[stab_index] = new_stab

        # add new stabilizers
        for new_photon in range(new_photons):
            new_stab = PauliString("I"*(self.n_qubits + new_photons + new_memories + new_emitters))
            new_stab[self.n_photons + new_photon] = "Z"
            self.stab_list.append(new_stab)
        for new_memory in range(new_memories):
            new_stab = PauliString("I"*(self.n_qubits + new_photons + new_memories + new_emitters))
            new_stab[self.n_photons + new_photons + self.n_memories + new_memory] = "Z"
            self.stab_list.append(new_stab)
        for new_emitter in range(new_emitters):
            new_stab = PauliString("I"*(self.n_qubits + new_photons + new_memories + new_emitters))
            new_stab[self.n_photons + new_photons + self.n_memories + new_memories + self.n_emitters + new_emitter] = "Z"
            self.stab_list.append(new_stab)

        # adjust qubit counts
        self.n_photons += new_photons
        self.n_memories += new_memories
        self.n_emitters += new_emitters
    
    def delete_qubit(self, qubit: int):
        # find stabilizer that acts on qubit
        qubit_stab_index = None
        qubit_pauli = None
        for stab_index in self.qubit_list:
            support = self.get_support(stab_index)
            if qubit in support:
                qubit_stab_index = stab_index
                qubit_pauli = self.get_pauli(stab_index, qubit)
                break
        if qubit_stab_index is None:
            raise Exception("Qubit not supported by any stabilizers")
        
        # remove qubit from other stabilizers
        for stab_index in self.qubit_list:
            if stab_index == qubit_stab_index:
                continue

            support = self.get_support(stab_index)
            if qubit in support:
                pauli = self.get_pauli(stab_index, qubit)
                if pauli != qubit_pauli:
                    raise Exception(f"Qubit {qubit} not disentangled")
                # multiply qubit out
                self.stab_mul(stab_index, qubit_stab_index)
        
            # delete qubit entry in stabilizer
            new_stab = self.stab_list[stab_index][:qubit] + self.stab_list[stab_index][qubit+1:]
            new_stab *= self.stab_list[stab_index].sign
            self.stab_list[stab_index] = new_stab

        # delete qubit stabilizer
        del self.stab_list[qubit_stab_index]

        # adjust qubit numbers
        if qubit in self.photon_list:
            self.n_photons -= 1
        elif qubit in self.memory_list:
            self.n_memories -= 1
        elif qubit in self.emitter_list:
            self.n_emitters -= 1
        
    def delete_qubits(self, qubits: list[int]):
        for qubit in qubits:
            self.delete_qubit(qubit)

    def pprint(self):
        result = ""
        digits = int(np.ceil(np.log10(self.n_qubits)))
        for index, stab in enumerate(self.stab_list):
            result = result + f"{index:0{digits}}: "
            if stab.sign == 1:
                result = result + "+"
            else:
                result = result + "-"

            for pauli in stab:
                if pauli == 1:
                    result = result + "\033[91mX\033[0m"
                elif pauli == 2:
                    result = result + "\033[92mY\033[0m"
                elif pauli == 3:
                    result = result + "\033[94mZ\033[0m"
                else:
                    result = result + "_"

            result = result + "\n"

        print(result)

    def stab_swap(self, stab1: int, stab2: int):
        if stab1 not in range(self.n_qubits):
            raise Exception(f"Invalid stabilizer: {stab1}")
        if stab2 not in range(self.n_qubits):
            raise Exception(f"Invalid stabilizer: {stab2}")

        self.stab_list[stab1], self.stab_list[stab2] = self.stab_list[stab2], self.stab_list[stab1]

    def stab_mul(self, stab1: int, stab2: int):
        if stab1 not in range(self.n_qubits):
            raise Exception(f"Invalid stabilizer: {stab1}")
        if stab2 not in range(self.n_qubits):
            raise Exception(f"Invalid stabilizer: {stab2}")
        if stab1 == stab2:
            raise Exception(f"Multiplying stabilizer {stab1} by self results in identity")

        self.stab_list[stab1] = self.stab_list[stab1] * self.stab_list[stab2]

    def clifford(self, gate: str, qubit1: int, qubit2: int = None):
        if qubit1 not in range(self.n_qubits):
            raise Exception(f"Invalid qubit: {qubit1}")
        if qubit2 is not None and qubit2 not in range(self.n_qubits):
            raise Exception(f"Invalid qubit: {qubit2}")

        if qubit1 == qubit2:
            raise Exception(f"Qubits are identical: {qubit1}")

        tab = Tableau.from_named_gate(gate)

        for index, stab in enumerate(self.stab_list):
            if qubit2 is None:
                self.stab_list[index] = stab.after(tab, [qubit1])
            else:
                self.stab_list[index] = stab.after(tab, [qubit1, qubit2])

    def rref(self):
        upper_stab = 0
        left_qubit = 0
        while left_qubit < self.n_qubits and upper_stab < self.n_qubits:
            num_stabs = 0
            stab1 = self.n_qubits
            stab2 = self.n_qubits

            # find stabilizer, stab1, that acts on left_qubit
            for stab in range(upper_stab, self.n_qubits):
                if self.get_pauli(stab, left_qubit) != "I":
                    stab1 = stab
                    num_stabs += 1
                    break

            # find stabilizer, stab2, that acts on left_qubit differently from stab1
            for stab in range(stab1, self.n_qubits):
                if self.get_pauli(stab, left_qubit) != "I":
                    if self.get_pauli(stab, left_qubit) != self.get_pauli(stab1, left_qubit):
                        stab2 = stab
                        num_stabs += 1
                        break

            # if no stabilizers act on left_qubit, move onto next qubit
            if num_stabs == 0:
                left_qubit += 1

            # if one stabilizer acts on left_qubit uniquely, move it to upper_stab. multiply it to
            # other stabilizers such that if upper_stab acts as X then all other stabilizers act only as
            # Z, and if upper_stab acts as Y or Z then all other stabilizers act as X
            elif num_stabs == 1:
                self.stab_swap(upper_stab, stab1)
                for stab in range(self.n_qubits):
                    pauli = self.get_pauli(stab, left_qubit)
                    upper_pauli = self.get_pauli(upper_stab, left_qubit)
                    if stab == upper_stab:
                        pass
                    elif pauli == upper_pauli or (pauli, upper_pauli) in {('Y', 'X'), ('Z', 'Y'), ('Y', 'Z')}:
                        self.stab_mul(stab, upper_stab)
                upper_stab += 1
                left_qubit += 1

            # if two stabilizers act on left_qubit, move them such that upper_stab acts as X and
            # upper_stab+1 acts as Z, and all other stabilizers act as identity
            elif num_stabs == 2:
                self.stab_swap(upper_stab, stab1)
                self.stab_swap(upper_stab+1, stab2)
                if self.get_pauli(upper_stab, left_qubit) == "Z": # make upper_stab act as X or Y
                    self.stab_swap(upper_stab, upper_stab+1)
                if self.get_pauli(upper_stab+1, left_qubit) != "Z": # make upper_stab+1 act as Z
                    self.stab_mul(upper_stab+1, upper_stab)
                if self.get_pauli(upper_stab, left_qubit) != "X": # make upper_stab act as X
                    self.stab_mul(upper_stab, upper_stab+1)
                for stab in range(self.n_qubits): # cancel all other stabs
                    if stab in {upper_stab, upper_stab+1}:
                        continue
                    if self.get_pauli(stab, left_qubit) in {"X", "Y"}:
                        self.stab_mul(stab, upper_stab)
                    if self.get_pauli(stab, left_qubit) in {"Y", "Z"}:
                        self.stab_mul(stab, upper_stab+1)
                left_qubit += 1
                upper_stab += 2

    def back_substitution(self, stabs: list[int] = None):
        if stabs is None:
            stabs = self.qubit_list
        for index, chosen_stab in enumerate(stabs[::-1]):
            for other_stab in stabs[:-index-1]:
                other_stab_weight = len(self.get_support(other_stab))

                new_stab = self.stab_list[chosen_stab] * self.stab_list[other_stab]
                new_stab_weight = len(new_stab.pauli_indices())

                if new_stab_weight < other_stab_weight:
                    self.stab_list[other_stab] = new_stab

    def height_function(self) -> list[int]:
        self.rref()

        leftmost_qubits = []
        for stab in range(self.n_qubits):
            for qubit in range(self.n_qubits):
                if self.get_pauli(stab, qubit) != "I":
                    leftmost_qubits.append(qubit+1)
                    break

        height = []
        for qubit in range(self.n_qubits + 1):
            num_stabs = sum(leftmost > qubit for leftmost in leftmost_qubits)
            height.append(self.n_qubits - qubit - num_stabs)

        return height

    def get_support(self, stab: int) -> list[int]:
        if stab not in self.qubit_list:
            raise Exception(f"Invalid stabilizer: {stab}")

        return self.stab_list[stab].pauli_indices()

    def get_pauli(self, stab: int, qubit: int) -> str:
        if stab not in self.qubit_list:
            raise Exception(f"Invalid stabilizer: {stab}")
        if qubit not in self.qubit_list:
            raise Exception(f"Invalid qubit: {qubit}")
        
        pauli_num = self.stab_list[stab][qubit]
        
        return ["I", "X", "Y", "Z"][pauli_num]
      
    def get_graph_flex(self) -> tuple[Graph, Protocol]:
        protocol = []
        for qubit in self.qubit_list:
            for stab in range(qubit, self.n_qubits):
                if self.get_pauli(stab, qubit) != "I":
                    self.stab_swap(qubit, stab)
                    break

            pauli = self.get_pauli(qubit, qubit)
            if pauli == "Y":
                self.clifford("S", qubit)
                protocol.append(("S", qubit))
            elif pauli == "Z":
                self.clifford("H", qubit)
                protocol.append(("H", qubit))

            for stab in self.qubit_list:
                if stab == qubit:
                    continue
                if self.get_pauli(stab, qubit) in {"X", "Y"}:
                    self.stab_mul(stab, qubit)

        for qubit in self.qubit_list:
            pauli = self.get_pauli(qubit, qubit)
            if pauli == "Y":
                self.clifford("S", qubit)
                protocol.append(("S", qubit))

            if self.stab_list[qubit].sign == -1:
                self.clifford("Z", qubit)
                protocol.append(("Z", qubit))

        edges = []
        for qubit1 in self.qubit_list:
            for qubit2 in range(qubit1+1, self.n_qubits):
                if self.get_pauli(qubit1, qubit2) == "Z":
                    edges.append((qubit1, qubit2))

        return Graph(n_photons=self.n_photons, n_memories=self.n_memories, n_emitters=self.n_emitters, edges=edges), protocol
    
    def get_graph_strict(self) -> Graph:
        state = self.copy()

        for qubit in state.qubit_list:
            x_stab_found = False
            for stab in range(qubit, state.n_qubits):
                if state.get_pauli(stab, qubit) in {"X", "Y"}:
                    x_stab_found = True
                    state.stab_swap(qubit, stab)
                    break
            if not x_stab_found:
                return None
            
            for stab in state.qubit_list:
                if stab == qubit:
                    continue
                if state.get_pauli(stab, qubit) in {"X", "Y"}:
                    state.stab_mul(stab, qubit)

        for qubit in state.qubit_list:
            if state.get_pauli(qubit, qubit) == "Y":
                return None
            if state.stab_list[qubit].sign == -1:
                return None
            
        self.stab_list = state.stab_list

        edges = []
        for qubit1 in self.qubit_list:
            for qubit2 in range(qubit1+1, self.n_qubits):
                if self.get_pauli(qubit1, qubit2) == "Z":
                    edges.append((qubit1, qubit2))

        return Graph(n_photons=self.n_photons, n_memories=self.n_memories, n_emitters=self.n_emitters, edges=edges)
    
    def force_measure(self, measurement: str | PauliString) -> bool:
        if type(measurement) == str:
            measurement = PauliString(measurement)
        if len(measurement) != self.n_qubits:
            raise Exception(f"Expected measurement of length {self.n_qubits}, got length {len(measurement)}")

        anticommuting = []

        for index, stab in enumerate(self.stab_list):
            if not measurement.commutes(stab):
                anticommuting.append(index)
        
        if len(anticommuting) == 0:
            # write measurement in basis of the generators, replace the one with the maximum weight
            matrix = GF2(np.column_stack([pauli_to_vector(stab) for stab in self.stab_list]))
            vector = GF2(pauli_to_vector(measurement)).reshape(-1, 1)
            augmented_matrix = np.hstack((matrix, vector))
            rref_matrix = augmented_matrix.row_reduce()
            solution = rref_matrix[:matrix.shape[1], -1]
            component_stabs = np.nonzero(np.array(solution))[0]

            chosen_stab = None
            chosen_weight = -np.inf
            for stab in component_stabs:
                weight = len(self.get_support(stab))
                if weight > chosen_weight:
                    chosen_stab = stab
                    chosen_weight = weight

            new_stab = PauliString("I"*self.n_qubits)
            for stab in component_stabs:
                new_stab *= self.stab_list[stab]

            self.stab_list[chosen_stab] = new_stab

            return False
        
        replace_index = anticommuting[0]
        for other_index in anticommuting[1:]:
            self.stab_mul(other_index, replace_index)

        self.stab_list[replace_index] = measurement

        return True
    
    def measurement_byproduct(self, measurement: str | PauliString) -> PauliString:
        if type(measurement) == str:
            measurement = PauliString(measurement)
        if len(measurement) != self.n_qubits:
            raise Exception(f"Expected measurement of length {self.n_qubits}, got length {len(measurement)}")

        anticommuting = []

        for index, stab in enumerate(self.stab_list):
            if not measurement.commutes(stab):
                anticommuting.append(index)
        
        if len(anticommuting) == 0:
            return None
        
        byproduct = None
        byproduct_weight = np.inf
        for index in anticommuting:
            stab = self.stab_list[index]
            if len(stab) < byproduct_weight:
                byproduct = stab
                byproduct_weight = len(stab)

        return byproduct

    
    def reset(self, qubit: int) -> bool:
        if qubit not in self.qubit_list:
            raise Exception(f"Unexpected qubit {qubit}")
        
        measurement = PauliString("I"*self.n_qubits)
        measurement[qubit] = "Z"

        result = self.force_measure(measurement)

        if -measurement in self.stab_list:
            self.clifford("X", qubit)

        return result


    def apply_protocol(self, protocol: Protocol):
        for step in protocol:
            if step[0] == "R":
                measurement = "_"*step[1] + "Z" + "_"*(self.n_qubits - step[1] - 1)
                self.force_measure(measurement)
            elif step[0] == "MR":
                for feedforward in step[2]:
                    self.clifford("C" + feedforward[0], step[1], feedforward[1])
                measurement = "_"*step[1] + "Z" + "_"*(self.n_qubits - step[1] - 1)
                self.force_measure(measurement)
            elif step[0] in {"RX Error", "RY Error", "RZ Error", "MR"}:
                raise Exception(f"Unexpected operation: {step[0]}")
            elif len(step) == 2:
                self.clifford(step[0], step[1])
            elif len(step) == 3:
                self.clifford(step[0], step[1], step[2])


    def get_sign(self, stab: int) -> int:
        return self.stab_list[stab].sign