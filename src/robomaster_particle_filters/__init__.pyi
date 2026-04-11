from . import plate_orbit
from . import fast_plate_orbit
from . import fast_plate_orbit_with_z_offset
from . import plate_orbit_v2

def get_include_dirs() -> list[str]: ...
def is_header_only_install() -> bool: ...
