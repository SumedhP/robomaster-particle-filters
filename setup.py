import os

from skbuild import setup

__version__ = "1.0.8"

base_setup_options = {
    "name": "robomaster_particle_filters",
    "version": __version__,
    "author": "Connor McMonigle",
    "author_email": "connormcmonigle@gmail.com",
    "description": "A native module implementing CUDA accelerated particle filters for the RoboMaster robotics competition.",
    "long_description": "",
    "zip_safe": False,
    "packages": ['robomaster_particle_filters'],
    "package_data": {'robomaster_particle_filters': ['*.pyi', 'py.typed']},
    "package_dir": {'': 'src'},
    "python_requires": ">=3.8",
}

additional_native_setup_options = {
    "cmake_install_dir": 'src/robomaster_particle_filters',
}

header_only = os.getenv("ROBOMASTER_PF_HEADER_ONLY", "0").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}

cmake_args = [
    f"-DROBOMASTER_PF_HEADER_ONLY={'ON' if header_only else 'OFF'}",
]

additional_native_setup_options["cmake_args"] = cmake_args

setup(
    **base_setup_options,
    **additional_native_setup_options,
)
