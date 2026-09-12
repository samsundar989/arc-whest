"""Lock and bind a bridge CLI's complete source graph before importing it.

This file is intentionally stdlib-only.  It reads a literal path registry from
the target's source with ``ast`` while the target is locked, then holds every
declared file stable while it compiles the target bytes and runs ``main``.
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import ctypes
import hashlib
import importlib.util
import os
import re
import sys
import tempfile
import types


_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
_SELF_FINGERPRINT = "b66ac9b57df160b363ffd1d49e8f372088ef5046442d107cff8b974ea86ade3d"
_REGISTRY_NAME = "_CLI_SOURCE_REGISTRY"
_TARGETS = {"p2_freeze_k4_arms.py", "p2_lab.py"}


class BootstrapRefused(ValueError):
    pass


def _sha256_bytes(raw):
    return hashlib.sha256(raw).hexdigest()


def _read_bytes(path, label):
    try:
        with open(path, "rb") as fh:
            raw = fh.read()
    except OSError as exc:
        raise BootstrapRefused("cannot read %s: %s" % (label, exc)) from exc
    return raw, _sha256_bytes(raw)


def _normalized_self_sha(raw):
    marker = re.compile(rb'(?m)^_SELF_FINGERPRINT = "[0-9a-f]{64}"')
    normalized, count = marker.subn(
        b'_SELF_FINGERPRINT = "' + (b'0' * 64) + b'"', raw)
    if count != 1:
        raise BootstrapRefused(
            "bootstrap source fingerprint marker is missing or ambiguous")
    return _sha256_bytes(normalized.replace(b"\r\n", b"\n"))


def _assert_loaded_source():
    raw, exact_sha = _read_bytes(os.path.abspath(__file__), "bootstrap source")
    if _normalized_self_sha(raw) != _SELF_FINGERPRINT:
        raise BootstrapRefused(
            "loaded bootstrap source does not match the locked disk snapshot")
    return exact_sha


def _repo_path(raw, label):
    if not isinstance(raw, str) or not raw:
        raise BootstrapRefused("%s must be a nonempty path string" % label)
    path = os.path.realpath(os.path.abspath(
        raw if os.path.isabs(raw) else os.path.join(_ROOT, raw)))
    try:
        inside = (os.path.commonpath([os.path.realpath(_ROOT), path]) ==
                  os.path.realpath(_ROOT))
    except ValueError:
        inside = False
    if not inside:
        raise BootstrapRefused("%s escapes the repository" % label)
    return path


def _relative(path):
    return os.path.relpath(os.path.abspath(path), _ROOT).replace("\\", "/")


def _same_path(left, right):
    return (os.path.normcase(os.path.realpath(left)) ==
            os.path.normcase(os.path.realpath(right)))


def _inside(path, root):
    try:
        return os.path.commonpath([path, root]) == root
    except ValueError:
        return False


def _external_import_roots():
    """Return environment package roots that may physically live in the repo."""
    roots = []
    for raw in sys.path:
        if not raw:
            continue
        path = os.path.normcase(os.path.realpath(os.path.abspath(raw)))
        parts = {part.lower() for part in path.split(os.sep)}
        if "site-packages" in parts or "dist-packages" in parts:
            roots.append(path)
    return tuple(roots)


def _repo_origin_candidates(origin, external_roots):
    """Map an import origin to lockable repo files, including cached bytecode."""
    if not isinstance(origin, str) or origin in {"built-in", "frozen"}:
        return ()
    path = os.path.normcase(os.path.realpath(os.path.abspath(origin)))
    root = os.path.normcase(os.path.realpath(_ROOT))
    if not _inside(path, root) or any(_inside(path, ext) for ext in external_roots):
        return ()
    candidates = {path}
    if path.endswith((".pyc", ".pyo")):
        try:
            source = importlib.util.source_from_cache(path)
        except ValueError:
            source = path[:-1]
        candidates.add(os.path.normcase(os.path.realpath(source)))
    return tuple(candidates)


class _RegistryImportGuard:
    """Resolve imports normally but reject unbound repo source before loading."""

    def __init__(self, finders, allowed_paths):
        self._finders = tuple(finders)
        self._allowed = {
            os.path.normcase(os.path.realpath(os.path.abspath(path)))
            for path in allowed_paths
        }
        self._external = _external_import_roots()
        self.violation = None

    def _check_spec(self, fullname, spec):
        candidates = _repo_origin_candidates(spec.origin, self._external)
        if candidates and not self._allowed.intersection(candidates):
            self.violation = (
                "repo-local import %s resolves outside the locked registry: %s" %
                (fullname, spec.origin))
            raise BootstrapRefused(self.violation)

    def find_spec(self, fullname, path=None, target=None):
        for finder in self._finders:
            method = getattr(finder, "find_spec", None)
            if method is None:
                continue
            spec = method(fullname, path, target)
            if spec is not None:
                self._check_spec(fullname, spec)
                return spec
        return None


def _purge_repo_modules_for_guard(allowed_paths):
    """Remove cached repo modules so every target import passes the guard."""
    external = _external_import_roots()
    bootstrap = os.path.normcase(os.path.realpath(os.path.abspath(__file__)))
    for name, module in tuple(sys.modules.items()):
        spec = getattr(module, "__spec__", None)
        origin = getattr(spec, "origin", None) or getattr(module, "__file__", None)
        candidates = _repo_origin_candidates(origin, external)
        if candidates and bootstrap not in candidates:
            sys.modules.pop(name, None)


def _assert_loaded_repo_modules_registered(allowed_paths):
    """Catch loaders that bypassed meta_path but registered a local module."""
    allowed = {
        os.path.normcase(os.path.realpath(os.path.abspath(path)))
        for path in allowed_paths
    }
    external = _external_import_roots()
    for name, module in tuple(sys.modules.items()):
        spec = getattr(module, "__spec__", None)
        origin = getattr(spec, "origin", None) or getattr(module, "__file__", None)
        candidates = _repo_origin_candidates(origin, external)
        if candidates and not allowed.intersection(candidates):
            raise BootstrapRefused(
                "loaded repo-local module %s is absent from the locked registry: %s" %
                (name, origin))


@contextlib.contextmanager
def _stable_files(paths):
    unique, seen = [], set()
    for raw in paths:
        path = os.path.normcase(os.path.realpath(os.path.abspath(raw)))
        if path not in seen:
            seen.add(path)
            unique.append(path)
    if os.name == "nt":
        from ctypes import wintypes
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        create = kernel32.CreateFileW
        create.argtypes = (wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD,
                           wintypes.LPVOID, wintypes.DWORD, wintypes.DWORD,
                           wintypes.HANDLE)
        create.restype = wintypes.HANDLE
        close = kernel32.CloseHandle
        close.argtypes = (wintypes.HANDLE,)
        handles = []
        try:
            for path in unique:
                handle = create(path, 0x80000000, 0x00000001, None, 3,
                                0x00000080, None)
                if handle == wintypes.HANDLE(-1).value:
                    raise BootstrapRefused(
                        "cannot lock bootstrap dependency %s: WinError %d" %
                        (path, ctypes.get_last_error()))
                handles.append(handle)
            yield
        finally:
            for handle in reversed(handles):
                close(handle)
        return
    import fcntl
    files = []
    try:
        for path in unique:
            fh = open(path, "rb")
            files.append(fh)
            fcntl.flock(fh.fileno(), fcntl.LOCK_SH | fcntl.LOCK_NB)
        yield
    except (OSError, BlockingIOError) as exc:
        raise BootstrapRefused(
            "cannot lock bootstrap dependencies: %s" % exc) from exc
    finally:
        for fh in reversed(files):
            try:
                fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
            finally:
                fh.close()


def _literal_registry(target_raw, target):
    try:
        tree = ast.parse(target_raw.decode("utf-8"), filename=target)
    except (UnicodeDecodeError, SyntaxError) as exc:
        raise BootstrapRefused("target source does not parse: %s" % exc) from exc
    values = []
    for node in tree.body:
        if (isinstance(node, ast.Assign) and len(node.targets) == 1 and
                isinstance(node.targets[0], ast.Name) and
                node.targets[0].id == _REGISTRY_NAME):
            try:
                values.append(ast.literal_eval(node.value))
            except (ValueError, TypeError) as exc:
                raise BootstrapRefused(
                    "target source registry is not literal") from exc
    if len(values) != 1:
        raise BootstrapRefused(
            "target source must declare exactly one literal %s" % _REGISTRY_NAME)
    registry = values[0]
    if (not isinstance(registry, dict) or not registry or
            any(not isinstance(name, str) or not name or
                not isinstance(path, str) or not path
                for name, path in registry.items())):
        raise BootstrapRefused("target source registry is invalid")
    paths = {name: _repo_path(path, "target source %s" % name)
             for name, path in registry.items()}
    if paths.get("locked_bootstrap") is None or not _same_path(
            paths["locked_bootstrap"], os.path.abspath(__file__)):
        raise BootstrapRefused("target does not bind this bootstrap source")
    target_names = [name for name, path in paths.items()
                    if _same_path(path, target)]
    if len(target_names) != 1:
        raise BootstrapRefused("target source registry does not bind the target once")
    return paths, target_names[0]


def _binding_snapshot(paths):
    return {name: {"path": _relative(path),
                   "sha256": _read_bytes(path, "source %s" % name)[1]}
            for name, path in paths.items()}


def _execute_target(target, target_raw, paths, target_name, bindings, args):
    _purge_repo_modules_for_guard(paths.values())
    with tempfile.TemporaryDirectory(prefix="p2-locked-pycache-") as cache:
        previous_prefix = sys.pycache_prefix
        sys.pycache_prefix = cache
        try:
            module = types.ModuleType("_arc_whest_locked_target")
            module.__file__ = target
            module.__package__ = None
            module.__builtins__ = __builtins__
            original_meta_path = list(sys.meta_path)
            guard = _RegistryImportGuard(original_meta_path, paths.values())
            sys.meta_path[:] = [guard]
            try:
                exec(compile(target_raw, target, "exec"), module.__dict__)
                if guard.violation is not None:
                    raise BootstrapRefused(guard.violation)
                _assert_loaded_repo_modules_registered(paths.values())
                if (not hasattr(module, "_cli_source_paths") or
                        not hasattr(module, "main")):
                    raise BootstrapRefused("target lacks the locked CLI interface")
                actual = module._cli_source_paths()
                if (not isinstance(actual, dict) or set(actual) != set(paths) or
                        any(not _same_path(actual[name], paths[name])
                            for name in paths)):
                    raise BootstrapRefused(
                        "executed target source graph differs from its prelocked registry")
                try:
                    module._verify_cli_bindings(bindings, paths)
                except Exception as exc:
                    raise BootstrapRefused(
                        "executed target rejected prelocked bindings: %s" % exc
                    ) from exc
                if bindings[target_name]["sha256"] != _sha256_bytes(target_raw):
                    raise BootstrapRefused("executed target bytes differ from binding")
                module._BOUND_CLI_BINDINGS = bindings
                previous_argv0 = sys.argv[0]
                sys.argv[0] = target
                try:
                    result = int(module.main(args))
                finally:
                    sys.argv[0] = previous_argv0
                    module._BOUND_CLI_BINDINGS = None
                if guard.violation is not None:
                    raise BootstrapRefused(guard.violation)
                _assert_loaded_repo_modules_registered(paths.values())
                return result
            finally:
                sys.meta_path[:] = original_meta_path
        finally:
            sys.pycache_prefix = previous_prefix


def run(target, expected_target_sha, args):
    target = _repo_path(target, "bootstrap target")
    if os.path.basename(target) not in _TARGETS:
        raise BootstrapRefused("unsupported bootstrap target")
    if (not isinstance(expected_target_sha, str) or
            re.fullmatch(r"[0-9a-f]{64}", expected_target_sha) is None):
        raise BootstrapRefused("expected target SHA-256 is invalid")
    bootstrap = os.path.abspath(__file__)
    # Stage one binds the only code that executes before the complete graph is
    # known: this stdlib-only bootstrap and the target text parsed by ast.
    with _stable_files([bootstrap, target]):
        _assert_loaded_source()
        target_raw, target_sha = _read_bytes(target, "bootstrap target")
        if target_sha != expected_target_sha:
            raise BootstrapRefused(
                "target changed between the public launcher and bootstrap lock")
        paths, target_name = _literal_registry(target_raw, target)
        # Stage two holds the complete literal graph through import and main.
        with _stable_files(paths.values()):
            if _read_bytes(target, "bootstrap target")[1] != target_sha:
                raise BootstrapRefused("target changed while its graph was resolved")
            bindings = _binding_snapshot(paths)
            return _execute_target(
                target, target_raw, paths, target_name, bindings, args)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True)
    parser.add_argument("--expected-target-sha", required=True)
    parser.add_argument("args", nargs=argparse.REMAINDER)
    parsed = parser.parse_args(argv)
    args = list(parsed.args)
    if args[:1] == ["--"]:
        args = args[1:]
    try:
        return run(parsed.target, parsed.expected_target_sha, args)
    except BootstrapRefused as exc:
        print("REFUSED: %s" % exc)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
