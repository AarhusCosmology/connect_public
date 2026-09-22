import os
import sys
from pathlib import Path
from types import ModuleType

import tensorflow as tf
import tf_keras

FILE_PATH = os.path.realpath(os.path.dirname(__file__))
CONNECT_PATH = Path(FILE_PATH).parents[4]

"""
This module provides a wrapper class 'Class' that mimics the interface of the CLASS Boltzmann solver,
allowing users to load and interact with trained CONNECT models seamlessly.

This is a placeholder module that gets morphed into the actual wrapper class for neural networks trained
with the specific cosmic solver (CLASS or CAMB, natively) when a model is loaded.
"""

class CosmoComputationError(Exception):
    pass

class CosmoSevereError(Exception):
    pass

class Class():
    """
    Placeholder / proxy that delegates to a backend implementation stored in self._impl.
    """

    def __init__(self, input_parameters=None, model_name=None):
        self._impl = None
        self.pars = dict(input_parameters or {})
        self._model_name = model_name
        self.model = None
        self.info = None

        if 'connect_model' in self.pars and (self._model_name is None):
            self.model_name = self.pars['connect_model']

    @property
    def model_name(self):
        return self._model_name

    @model_name.setter
    def model_name(self, new_name):
        self._model_name = new_name
        # defer loading until we actually need to
        if self._impl is None:
            self._load_model_and_morph_class(model_name=new_name)

    def set(self, *args, **kwargs):
        # Merge parameters
        if args:
            try:
                if len(args) == 1 and isinstance(args[0], dict):
                    self.pars.update(args[0])
                else:
                    self.pars.update(dict(*args))
            except Exception:
                pass
        self.pars.update(kwargs)

        if 'connect_model' in self.pars and (self._model_name is None):
            self.model_name = self.pars['connect_model']

        # Forward to backend if already present
        if self._impl is not None and hasattr(self._impl, 'set'):
            try:
                if args or kwargs:
                    self._impl.set(*args, **kwargs)
                else:
                    self._impl.set(self.pars)
            except TypeError:
                try:
                    self._impl.set(self.pars)
                except Exception:
                    pass
            except Exception:
                # Let backend exceptions propagate (they will be patched)
                raise

    def _load_model_and_morph_class(self, model_name=None):
        """
        Load the TF model, determine backend (camb/class), instantiate the correct backend wrapper,
        ensure class_module.Class.__init__ ran (so compute_class_background etc exist), patch exceptions,
        and set self._impl to the implementation instance.
        """
        if model_name is None:
            if self._model_name is None:
                raise NameError('No model was specified - Set the attribute model_name to the name of a trained CONNECT model')
            name = self._model_name
        else:
            name = model_name

        # Load keras model (try trained_models path then raw name)
        try:
            try:
                self.model = tf_keras.models.load_model(os.path.join(CONNECT_PATH, 'trained_models', name), compile=False)
            except Exception:
                self.model = tf_keras.models.load_model(name, compile=False)
        except Exception:
            raise NameError(f"No trained model by the name of '{name}'")

        # Extract metadata
        try:
            self.info = eval(self.model.get_raw_info().numpy().decode('utf-8'))
        except Exception:
            raise ValueError('The loaded model does not seem to be a valid CONNECT model')

        # Make sure our local package path is importable
        if FILE_PATH not in sys.path:
            sys.path.insert(1, FILE_PATH)

        solver = self.info.get('cosmic_solver', '').lower()

        backend_class = None
        BackendCosmoComp = None
        BackendCosmoSev = None

        if solver == 'camb':
            try:
                import camb_module
                backend_class = camb_module.Class
                BackendCosmoComp = camb_module.CosmoComputationError
                BackendCosmoSev = camb_module.CosmoSevereError
            except Exception as e:
                raise ImportError(f"Failed to import CAMB wrapper from camb_module: {e}")

        elif solver == 'class':
            try:
                import class_module
                backend_class = class_module.Class
                BackendCosmoComp = class_module.CosmoComputationError
                BackendCosmoSev = class_module.CosmoSevereError
            except Exception as e:
                raise ImportError(f"Failed to import CLASS wrapper from class_module: {e}")
        else:
            raise ValueError(
                f"CONNECT model cosmic_solver '{self.info.get('cosmic_solver')}' not recognized. "
                "Supported solvers are 'camb' and 'class'."
            )

        impl = backend_class(self.pars)

        # Replace module-level exception objects with backend exceptions
        this_mod = sys.modules[__name__]
        old_comp = this_mod.CosmoComputationError
        old_sev = this_mod.CosmoSevereError
        this_mod.CosmoComputationError = BackendCosmoComp
        this_mod.CosmoSevereError = BackendCosmoSev

        # Patch other modules that imported the old placeholders
        self._patch_imported_exceptions(old_comp, BackendCosmoComp)
        self._patch_imported_exceptions(old_sev, BackendCosmoSev)

        # Set implementation
        self._impl = impl

        # Finally, forward parameters to impl.set if available
        if hasattr(self._impl, 'set'):
            try:
                self._impl.set(self.pars)
            except TypeError:
                try:
                    self._impl.set()
                except Exception:
                    pass

    def struct_cleanup(self):
        if self._impl is not None and hasattr(self._impl, 'struct_cleanup'):
            try:
                return self._impl.struct_cleanup()
            except Exception:
                pass
        return None

    def state(self):
        if self._impl is not None and hasattr(self._impl, 'state'):
            try:
                return self._impl.state()
            except Exception:
                pass
        return True

    def empty(self):
        if self._impl is not None and hasattr(self._impl, 'empty'):
            try:
                return self._impl.empty()
            except Exception:
                pass
        return None

    def _patch_imported_exceptions(self, old_exc_obj, new_exc_obj):
        """
        Replace references to old_exc_obj found in other loaded modules'
        attributes with new_exc_obj.
        """
        for mod_name, mod in list(sys.modules.items()):
            if not isinstance(mod, ModuleType):
                continue
            for attr_name, val in list(getattr(mod, '__dict__', {}).items()):
                if val is old_exc_obj:
                    mod.__dict__[attr_name] = new_exc_obj

    def __getattr__(self, name):
        impl = self.__dict__.get('_impl')
        if impl is None:
            raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'. Backend not loaded yet.")
        return getattr(impl, name)

    def __setattr__(self, name, value):
        internal = {'_impl', 'pars', '_model_name', 'model', 'info'}
        if name in internal:
            return object.__setattr__(self, name, value)
        impl = self.__dict__.get('_impl')
        if impl is not None and hasattr(impl, name):
            try:
                setattr(impl, name, value)
                return
            except Exception:
                pass
        return object.__setattr__(self, name, value)

    def __dir__(self):
        result = set(dir(type(self)))
        result.update(self.__dict__.keys())
        impl = self.__dict__.get('_impl')
        if impl is not None:
            try:
                result.update(dir(impl))
            except Exception:
                pass
        return sorted(result)

    def __repr__(self):
        impl = self.__dict__.get('_impl')
        if impl is None:
            return f"<Class (placeholder, uninitialized) model_name={self._model_name}>"
        return f"<Class delegating to {impl!r}>"
