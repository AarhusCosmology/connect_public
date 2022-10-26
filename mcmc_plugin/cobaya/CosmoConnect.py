import sys
import os
import numpy as np
from pathlib import Path

from cobaya.theories.classy import classy
from cobaya.component import ComponentNotInstalledError, load_external_module

class CosmoConnect(classy):

    def initialize(self):
        """Importing CONNECT from the correct path."""
        classy_path = Path(self.path).parents[0]
        sys.path.insert(1, self.get_import_path(classy_path))
        import classy as connect
        self.classy_module = connect
        self.classy = self.classy_module.Class()
        super(classy, self).initialize()
        self.extra_args["output"] = self.extra_args.get("output", "")
        self.derived_extra = []

    @staticmethod
    def get_import_path(path):
        source_path = os.path.join(path, "python")
        if not os.path.isdir(source_path):
            raise FileNotFoundError(f"Source path {source_path} not found.")
        build_path = os.path.join(source_path, "build")
        if not os.path.isdir(build_path):
            raise FileNotFoundError(f"`build` folder not found for source path {source_path}."
                                    f" Maybe compilation failed?")
        # Folder starts with `lib.`
        try:
            post = next(d for d in os.listdir(build_path) if d.startswith('lib.'))
        except StopIteration:
            raise FileNotFoundError(
                f"No `lib.[...]` folder found containing compiled products at {source_path}. ")
        return os.path.join(build_path, post)
