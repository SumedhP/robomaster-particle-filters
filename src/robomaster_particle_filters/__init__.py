import importlib
from pathlib import Path

_extension_import_error = None

try:
	_native_module = importlib.import_module(".robomaster_particle_filters", __name__)
	fast_plate_orbit = _native_module.fast_plate_orbit
	fast_plate_orbit_with_z_offset = _native_module.fast_plate_orbit_with_z_offset
	plate_orbit = _native_module.plate_orbit
	plate_orbit_v2 = _native_module.plate_orbit_v2
except ImportError as exc:  # pragma: no cover - only reached in header-only installs.
	_extension_import_error = exc


def get_include_dirs() -> list[str]:
	"""Return package-local include roots for downstream native builds."""
	package_root = Path(__file__).resolve().parent
	return [
		str(package_root / "include"),
		str(package_root / "deps" / "eigen"),
		str(package_root / "deps" / "particle-filter"),
	]


def is_header_only_install() -> bool:
	"""True when this wheel does not include the compiled extension module."""
	return _extension_import_error is not None


__all__ = ["get_include_dirs", "is_header_only_install"]

if _extension_import_error is None:
	__all__.extend([
		"plate_orbit",
		"fast_plate_orbit",
		"fast_plate_orbit_with_z_offset",
		"plate_orbit_v2",
	])
