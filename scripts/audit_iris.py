#!/usr/bin/env python3
"""Iris source code audit script.

Systematically audit the Iris codebase (github.com/ROCm/iris) to identify
core functions and patterns that XTile should reference or port.

Usage:
    python scripts/audit_iris.py [--repo-url URL] [--output-dir DIR]

Outputs:
    docs/iris_audit.json          - Structured audit data
    docs/iris_audit_summary.md    - Human-readable summary
"""

import argparse
import ast
import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class TritonFunction:
    """A @triton.jit decorated function found in the Iris codebase."""
    name: str
    file: str
    line: int
    params: list[str]
    num_lines: int
    docstring: str | None = None


@dataclass
class AtomicUsage:
    """A tl.atomic_* call site."""
    function: str
    file: str
    line: int
    atomic_op: str
    context_line: str


@dataclass
class IPCCall:
    """A hipIpc* or cudaIpc* call site."""
    function: str
    file: str
    line: int
    api_call: str
    context_line: str


@dataclass
class RemoteAccess:
    """A tl.load/tl.store used for remote memory access."""
    function: str
    file: str
    line: int
    op_type: str  # "load" or "store"
    context_line: str


@dataclass
class AuditReport:
    """Complete audit report."""
    repo_url: str
    commit_hash: str = ""
    triton_functions: list[TritonFunction] = field(default_factory=list)
    atomic_usages: list[AtomicUsage] = field(default_factory=list)
    ipc_calls: list[IPCCall] = field(default_factory=list)
    remote_accesses: list[RemoteAccess] = field(default_factory=list)
    key_functions: dict[str, str] = field(default_factory=dict)
    file_summary: dict[str, int] = field(default_factory=dict)


def clone_repo(repo_url: str, target_dir: Path) -> str:
    """Clone or update the Iris repository. Returns commit hash."""
    if target_dir.exists():
        print(f"Repository already exists at {target_dir}, pulling latest...")
        result = subprocess.run(
            ["git", "-C", str(target_dir), "pull"],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            print(f"Warning: git pull failed: {result.stderr}")
    else:
        print(f"Cloning {repo_url} to {target_dir}...")
        target_dir.parent.mkdir(parents=True, exist_ok=True)
        result = subprocess.run(
            ["git", "clone", "--depth", "1", repo_url, str(target_dir)],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            print(f"Error cloning repo: {result.stderr}")
            sys.exit(1)

    # Get commit hash
    result = subprocess.run(
        ["git", "-C", str(target_dir), "rev-parse", "HEAD"],
        capture_output=True, text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def find_python_files(root: Path) -> list[Path]:
    """Find all .py files in the repository."""
    return sorted(root.rglob("*.py"))


def analyze_file(filepath: Path, repo_root: Path) -> tuple[
    list[TritonFunction],
    list[AtomicUsage],
    list[IPCCall],
    list[RemoteAccess],
]:
    """Analyze a single Python file for Iris-relevant patterns."""
    triton_funcs = []
    atomic_usages = []
    ipc_calls = []
    remote_accesses = []

    rel_path = str(filepath.relative_to(repo_root))

    try:
        source = filepath.read_text(encoding="utf-8", errors="replace")
        lines = source.splitlines()
    except Exception as e:
        print(f"  Warning: Could not read {rel_path}: {e}")
        return triton_funcs, atomic_usages, ipc_calls, remote_accesses

    # Parse AST to find @triton.jit functions
    try:
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                is_triton_jit = any(
                    _is_triton_jit_decorator(d) for d in node.decorator_list
                )
                if is_triton_jit:
                    params = [arg.arg for arg in node.args.args]
                    end_line = getattr(node, "end_lineno", node.lineno + 10)
                    docstring = ast.get_docstring(node)
                    triton_funcs.append(TritonFunction(
                        name=node.name,
                        file=rel_path,
                        line=node.lineno,
                        params=params,
                        num_lines=end_line - node.lineno + 1,
                        docstring=docstring,
                    ))
    except SyntaxError:
        pass  # Skip files with syntax errors

    # Line-by-line scan for patterns
    for i, line in enumerate(lines, 1):
        stripped = line.strip()

        # Atomic operations
        if "tl.atomic_" in stripped or "atomic_" in stripped:
            for op in ["atomic_cas", "atomic_add", "atomic_xchg", "atomic_and",
                        "atomic_or", "atomic_xor", "atomic_min", "atomic_max"]:
                if op in stripped:
                    func_name = _find_enclosing_function(lines, i - 1)
                    atomic_usages.append(AtomicUsage(
                        function=func_name,
                        file=rel_path,
                        line=i,
                        atomic_op=op,
                        context_line=stripped,
                    ))
                    break

        # IPC calls
        if "hipIpc" in stripped or "cudaIpc" in stripped:
            func_name = _find_enclosing_function(lines, i - 1)
            api_call = _extract_api_call(stripped, ["hipIpc", "cudaIpc"])
            ipc_calls.append(IPCCall(
                function=func_name,
                file=rel_path,
                line=i,
                api_call=api_call,
                context_line=stripped,
            ))

        # Remote memory access patterns (tl.load/tl.store with translated pointers)
        if ("tl.load" in stripped or "tl.store" in stripped) and (
            "translate" in stripped or "remote" in stripped or "heap" in stripped
        ):
            op_type = "load" if "tl.load" in stripped else "store"
            func_name = _find_enclosing_function(lines, i - 1)
            remote_accesses.append(RemoteAccess(
                function=func_name,
                file=rel_path,
                line=i,
                op_type=op_type,
                context_line=stripped,
            ))

    return triton_funcs, atomic_usages, ipc_calls, remote_accesses


def _is_triton_jit_decorator(decorator: ast.expr) -> bool:
    """Check if a decorator is @triton.jit or similar."""
    if isinstance(decorator, ast.Attribute):
        if isinstance(decorator.value, ast.Name):
            return decorator.value.id == "triton" and decorator.attr == "jit"
    if isinstance(decorator, ast.Name):
        return decorator.id in ("jit", "triton_jit")
    return False


def _find_enclosing_function(lines: list[str], line_idx: int) -> str:
    """Find the name of the enclosing function for a given line."""
    for i in range(line_idx, -1, -1):
        stripped = lines[i].strip()
        if stripped.startswith("def "):
            name = stripped[4:].split("(")[0].strip()
            return name
    return "<module>"


def _extract_api_call(line: str, prefixes: list[str]) -> str:
    """Extract the API call name from a line."""
    for prefix in prefixes:
        idx = line.find(prefix)
        if idx >= 0:
            end = idx
            while end < len(line) and (line[end].isalnum() or line[end] == "_"):
                end += 1
            return line[idx:end]
    return "unknown"


def identify_key_functions(report: AuditReport) -> dict[str, str]:
    """Identify key functions for XTile to reference."""
    key_funcs = {}

    for func in report.triton_functions:
        name_lower = func.name.lower()

        if "translate" in name_lower or "xlat" in name_lower:
            key_funcs[func.name] = (
                f"Pointer translation function at {func.file}:{func.line} "
                f"({func.num_lines} lines). CRITICAL for XTile port."
            )

        if any(kw in name_lower for kw in ["load", "store", "get", "put", "copy"]):
            key_funcs[func.name] = (
                f"Data movement primitive at {func.file}:{func.line} "
                f"({func.num_lines} lines). Reference for communication primitives."
            )

        if any(kw in name_lower for kw in ["gemm", "matmul"]):
            key_funcs[func.name] = (
                f"GEMM kernel at {func.file}:{func.line} "
                f"({func.num_lines} lines). Reference for pattern implementations."
            )

        if any(kw in name_lower for kw in ["scatter", "gather", "reduce", "allreduce"]):
            key_funcs[func.name] = (
                f"Collective operation at {func.file}:{func.line} "
                f"({func.num_lines} lines). Reference for collective primitives."
            )

    return key_funcs


def generate_json_report(report: AuditReport, output_path: Path):
    """Save structured JSON report."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = asdict(report)
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"JSON report saved to {output_path}")


def generate_markdown_summary(report: AuditReport, output_path: Path):
    """Generate human-readable markdown summary."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Iris Source Code Audit Summary",
        "",
        f"**Repository**: {report.repo_url}",
        f"**Commit**: `{report.commit_hash[:12]}`",
        "",
        "---",
        "",
        "## Statistics",
        "",
        f"- **@triton.jit functions**: {len(report.triton_functions)}",
        f"- **Atomic operation sites**: {len(report.atomic_usages)}",
        f"- **IPC API calls**: {len(report.ipc_calls)}",
        f"- **Remote memory access sites**: {len(report.remote_accesses)}",
        "",
    ]

    # Triton functions table
    if report.triton_functions:
        lines.extend([
            "## @triton.jit Functions",
            "",
            "| Function | File | Line | Params | Lines |",
            "|----------|------|------|--------|-------|",
        ])
        for func in sorted(report.triton_functions, key=lambda f: f.file):
            params_str = ", ".join(func.params[:5])
            if len(func.params) > 5:
                params_str += ", ..."
            lines.append(
                f"| `{func.name}` | {func.file} | {func.line} | {params_str} | {func.num_lines} |"
            )
        lines.append("")

    # Key functions for XTile
    if report.key_functions:
        lines.extend([
            "## Key Functions for XTile",
            "",
        ])
        for name, desc in report.key_functions.items():
            lines.append(f"- **`{name}`**: {desc}")
        lines.append("")

    # Atomic operations
    if report.atomic_usages:
        lines.extend([
            "## Atomic Operations",
            "",
            "| Function | File | Line | Op | Context |",
            "|----------|------|------|----|---------|",
        ])
        for usage in report.atomic_usages[:50]:
            ctx = usage.context_line[:60] + "..." if len(usage.context_line) > 60 else usage.context_line
            lines.append(
                f"| `{usage.function}` | {usage.file} | {usage.line} | {usage.atomic_op} | `{ctx}` |"
            )
        lines.append("")

    # IPC calls
    if report.ipc_calls:
        lines.extend([
            "## IPC API Calls",
            "",
            "| Function | File | Line | API | Context |",
            "|----------|------|------|-----|---------|",
        ])
        for call in report.ipc_calls:
            ctx = call.context_line[:60] + "..." if len(call.context_line) > 60 else call.context_line
            lines.append(
                f"| `{call.function}` | {call.file} | {call.line} | {call.api_call} | `{ctx}` |"
            )
        lines.append("")

    # File summary
    if report.file_summary:
        lines.extend([
            "## Files Analyzed",
            "",
            "| File | Lines |",
            "|------|-------|",
        ])
        for file, num_lines in sorted(report.file_summary.items()):
            lines.append(f"| {file} | {num_lines} |")
        lines.append("")

    # Recommendations
    lines.extend([
        "## Recommendations for XTile",
        "",
        "### Must Port",
        "1. Pointer translation function (`__translate` or equivalent)",
        "2. Symmetric heap setup (IPC handle exchange flow)",
        "3. Value-based load/store primitives",
        "4. Pointer-based get/put/copy primitives",
        "",
        "### Should Reference",
        "1. Atomic operation patterns (sem/scope parameter handling)",
        "2. GEMM kernel structure (persistent kernel, swizzle)",
        "3. Overlap pattern implementations (bulk-sync, fused, WG-specialized)",
        "",
        "### Key Differences from Iris",
        "1. XTile adds NVIDIA backend (CUDA IPC vs HIP IPC)",
        "2. XTile adds Pattern Library abstraction",
        "3. XTile adds tile_signal/tile_wait (TileLink-inspired)",
        "4. XTile adds Auto-Select engine",
        "",
    ])

    with open(output_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Markdown summary saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Audit Iris source code for XTile development")
    parser.add_argument(
        "--repo-url",
        default="https://github.com/ROCm/iris.git",
        help="Iris repository URL (default: ROCm/iris)",
    )
    parser.add_argument(
        "--target-dir",
        default="vendor/iris",
        help="Directory to clone into (default: vendor/iris)",
    )
    parser.add_argument(
        "--output-dir",
        default="docs",
        help="Output directory for reports (default: docs)",
    )
    parser.add_argument(
        "--skip-clone",
        action="store_true",
        help="Skip cloning, use existing repo",
    )
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    target_dir = project_root / args.target_dir
    output_dir = project_root / args.output_dir

    # Step 1: Clone repository
    if not args.skip_clone:
        commit_hash = clone_repo(args.repo_url, target_dir)
    else:
        if not target_dir.exists():
            print(f"Error: {target_dir} does not exist. Remove --skip-clone.")
            sys.exit(1)
        result = subprocess.run(
            ["git", "-C", str(target_dir), "rev-parse", "HEAD"],
            capture_output=True, text=True,
        )
        commit_hash = result.stdout.strip() if result.returncode == 0 else "unknown"

    # Step 2: Find and analyze all Python files
    report = AuditReport(repo_url=args.repo_url, commit_hash=commit_hash)
    py_files = find_python_files(target_dir)
    print(f"\nFound {len(py_files)} Python files to analyze...")

    for filepath in py_files:
        rel_path = str(filepath.relative_to(target_dir))
        try:
            line_count = len(filepath.read_text(encoding="utf-8", errors="replace").splitlines())
            report.file_summary[rel_path] = line_count
        except Exception:
            continue

        funcs, atomics, ipcs, remotes = analyze_file(filepath, target_dir)
        report.triton_functions.extend(funcs)
        report.atomic_usages.extend(atomics)
        report.ipc_calls.extend(ipcs)
        report.remote_accesses.extend(remotes)

        if funcs or atomics or ipcs or remotes:
            print(f"  {rel_path}: {len(funcs)} triton funcs, {len(atomics)} atomics, "
                  f"{len(ipcs)} IPC calls, {len(remotes)} remote accesses")

    # Step 3: Identify key functions
    report.key_functions = identify_key_functions(report)

    # Step 4: Generate reports
    print(f"\n--- Audit Complete ---")
    print(f"@triton.jit functions: {len(report.triton_functions)}")
    print(f"Atomic operation sites: {len(report.atomic_usages)}")
    print(f"IPC API calls: {len(report.ipc_calls)}")
    print(f"Remote memory access sites: {len(report.remote_accesses)}")
    print(f"Key functions identified: {len(report.key_functions)}")

    generate_json_report(report, output_dir / "iris_audit.json")
    generate_markdown_summary(report, output_dir / "iris_audit_summary.md")

    print("\nDone! Review the reports in the docs/ directory.")


if __name__ == "__main__":
    main()
