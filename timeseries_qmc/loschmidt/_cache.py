

from abc import ABC, abstractmethod
import os
import pickle
import logging
import uuid
from pytket.extensions.qiskit import AerBackend

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class ShotsCache(ABC):
    def __init__(self, n_shots, backend, compilation_backend):
        self.n_shots = n_shots
        self.backend = backend
        if(compilation_backend is None):
            self.compilation_backend = backend
        else:
            self.compilation_backend = compilation_backend
        self.save_handles = not isinstance(self.backend, AerBackend)

    def get_handle(self, circuit):
        if not self.save_handles:
            compiled_circuit = self.get_compiled_circuit(circuit)
            return self.backend.process_circuit(compiled_circuit, n_shots=self.n_shots)

        handle = self._load_handle(circuit.name)
        if(handle is not None):
            return handle

        logger.info("Handle for '{}' not found. Submitting the circuit to the backend.".format(circuit.name))
        compiled_circuit = self.get_compiled_circuit(circuit)
        handle = self.backend.process_circuit(compiled_circuit, n_shots=self.n_shots)
        self._save_handle(handle, circuit.name)

        return handle
    
    def get_compiled_circuit(self, circuit):
        """ Retrieve the compiled version of the specified circuit. A circuit is identified by its 'name' attribute.
        
        If a compiled circuit of the same name exist, it is returned.
        Otherwise, the circuit is complied with the compilation backend
        """
        compiled_circuit = self._load_compiled_circuit(circuit.name)
        if(compiled_circuit is not None):
            return compiled_circuit

        logger.info("Compiled circuit for '{}' not found. Compiling the circuit.".format(circuit.name))
        compiled_circuit = self.compilation_backend.get_compiled_circuit(circuit)

        self._save_compiled_circuit(compiled_circuit, circuit.name)      
        return compiled_circuit

    def get_shots(self, circuit):
        """ Retrieve the shots of the specified circuit. A circuit is identified by its 'name' attribute.
        
        If the shots for a circuit of the same name exist, they are returned.
        Otherwise, if the handel of this name exists, its shots are retrieved from the backend, saved locally and returned.
        Otherwise, the circuit is submitted to the backend, waited for its result and then the shots are retrieved and returned.

        Note: When the shots are not ready, this method is bloked until they are. To avoid blocking, use 'prepare_shots' instead.
        """
        shots = self._load_shots(circuit.name)
        if(shots is not None):
            return shots


        logger.info("Shots for '{}' not found. Retrieving the result from the backend.".format(circuit.name))
        handle = self.get_handle(circuit)
        try:
            shots = self.backend.get_result(handle).get_shots()
        except Exception as e:
            logger.info("Following error occured while retrieving the result:\n"+str(e))
            logger.info("Removing the old job handle of {} and retrying.".format(circuit.name))
            self._remove_handle(circuit.name)
            return self.get_shots(circuit)
        
        self._save_shots(shots, circuit.name)          
        return shots

    def prepare_shots(self, circuit):
        """Notify the cache to prepare the shots for the specified circuit. A circuit is identified by its 'name' attribute.

        If neither the shots nor the handle for a circuit of the same name exist, 
        the circuit is submitted to the backend for processing and the handle is saved.

        Note: For AerBackend, handles are not saved. In that case, this method is superfluous.
        The reason for not saving AerBackend handles is that they become invalid after restarting.
        """
        if(self.save_handles):
            self.get_handle(circuit)

    @abstractmethod
    def _load_compiled_circuit(self, name):
        raise NotImplemented()
    
    @abstractmethod
    def _load_shots(self, name):
        raise NotImplemented()
    
    @abstractmethod
    def _load_handle(self, name):
        raise NotImplemented()
    
    @abstractmethod
    def _save_compiled_circuit(self, name):
        raise NotImplemented()
    
    @abstractmethod
    def _save_shots(self, name):
        raise NotImplemented()
    
    @abstractmethod
    def _save_handle(self, name):
        raise NotImplemented()

    @abstractmethod
    def _remove_handle(self, name):
        raise NotImplemented()
    
class FilesShotsCache(ShotsCache):
    """ Class for submitting circuit and storing their shots locally in a folder.

    Parameters
    ----------
    cache_directory: str
        Path to the directory where shots/handles are to be stored. 
        If ``None``, a local directory with random generated name is used.

    n_shots: int
        Number of shots to use when submitting the circuits.
    
    backend: Pytket Backend, optional
        Backend to use for processing the circuits.

    compilation_backend: Pytket Backend, optional
        Backend to use for compiling the circuits. Default value of ``None`` uses ``backend`` parameter for compilation.

    read_only: bool, optional
        If true, then only shots of existing circuits are retrieved. Asking for shots of new ciruits raises an error.   
    """
    def __init__(self, cache_directory, n_shots, backend = AerBackend(), compilation_backend = None, read_only = False):
        super().__init__(n_shots, backend, compilation_backend)
        
        if(cache_directory is None):
            cache_directory = str(uuid.uuid4()) # get a unique name

        self.read_only = read_only
        if(self.save_handles):
            self.handles_directory = os.path.join(cache_directory, "handles")
            if not self.read_only and not os.path.exists(self.handles_directory):
                os.makedirs(self.handles_directory)
        
        self.shots_directory = os.path.join(cache_directory, "shots")
           
        if not self.read_only and not os.path.exists(self.shots_directory):
            os.makedirs(self.shots_directory)

        self.circuits_directory = os.path.join(cache_directory, "circuits")
      
        if not self.read_only and not os.path.exists(self.circuits_directory):
            os.makedirs(self.circuits_directory)

    def _load_compiled_circuit(self, name):
        file_name = os.path.join(self.circuits_directory, name+".pickle")
        try:
            with open(file_name, "rb") as f:
                return pickle.load(f)
        except IOError:
            return None
    
    def _load_shots(self, name):
        file_name = os.path.join(self.shots_directory, name+".pickle")
        try:
            with open(file_name, "rb") as f:
                return pickle.load(f)
        except IOError:
            if(self.read_only):
                raise Exception("Shots for '{}' not found. Cannot proceed in read-only mode.".format(name))            
            return None
    
    def _load_handle(self, name):
        file_name = os.path.join(self.handles_directory, name+".pickle")
        try:
            with open(file_name, "rb") as f:
                return pickle.load(f)
        except IOError:
            return None
    
    def _save_compiled_circuit(self, compiled_circuit, name):
        file_name = os.path.join(self.circuits_directory, name+".pickle")
        with open(file_name, "wb") as f:
            pickle.dump(compiled_circuit, f)     
    
    def _save_shots(self, shots, name):
        file_name = os.path.join(self.shots_directory, name+".pickle")
        with open(file_name, "wb") as f:
            pickle.dump(shots, f)   
    
    def _save_handle(self, handle, name):
        file_name = os.path.join(self.handles_directory, name+".pickle")
        with open(file_name, "wb") as f:
            pickle.dump(handle, f)   

    def _remove_handle(self, name):
        file_name = os.path.join(self.handles_directory, name+".pickle")
        os.remove(file_name)

    def prepare_shots(self, circuit):
        if(not self.read_only):
            return super().prepare_shots(circuit)
        
class MemoryShotsCache(ShotsCache):
    """ Class for submitting circuit and storing their shots in temporarly in memory.

    Parameters
    ----------
    n_shots: int
        Number of shots to use when submitting the circuits.
    
    backend: Pytket Backend, optional
        Backend to use for processing the circuits.

    compilation_backend: Pytket Backend, optional
        Backend to use for compiling the circuits. Default value of ``None`` uses 'backend' parameter for compilation.
    """    
    def __init__(self, n_shots, backend = AerBackend(), compilation_backend = None):
        super().__init__(n_shots, backend, compilation_backend)

        if(self.save_handles):
            self.handles_dict = {}
        
        self.circuits_dict = {}
        self.shots_dict = {}

    def _load_compiled_circuit(self, name):
        return self.circuits_dict.get(name)

    def _load_shots(self, name):
        return self.shots_dict.get(name)
    
    def _load_handle(self, name):
        return self.handles_dict.get(name)
    
    def _save_compiled_circuit(self, compiled_circuit, name):
        self.circuits_dict[name] = compiled_circuit   
    
    def _save_shots(self, shots, name):
        self.shots_dict[name] = shots
    
    def _save_handle(self, handle, name):
        self.handles_dict[name] = handle  

    def _remove_handle(self, name):
        del self.handles_dict[name]