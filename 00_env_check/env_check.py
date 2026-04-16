"""
Module 00 — Environment Verification
Confirms that every package and configuration required for the workshop is present.
Run via env_check.sh before starting any other module.
"""
import sys
import os


def check(label, fn):
    try:
        result = fn()
        print(f"  [OK]   {label}: {result}")
        return True
    except Exception as e:
        print(f"  [FAIL] {label}: {e}")
        return False


def section(title):
    print(f"\n{'─'*55}")
    print(f"  {title}")
    print(f"{'─'*55}")


def try_import(name):
    import importlib
    m = importlib.import_module(name)
    return getattr(m, "__version__", "imported ok")


print("=" * 55)
print("  Distributed Parallel Deep Learning Workshop")
print("  Environment Verification")
print("=" * 55)

section("Python")
check("Python version", lambda: sys.version.split()[0])
check("Executable", lambda: sys.executable)

section("PyTorch & CUDA")
import torch
check("PyTorch version",    lambda: torch.__version__)
check("CUDA available",     lambda: torch.cuda.is_available())
check("CUDA device count",  lambda: torch.cuda.device_count())
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        check(f"GPU {i} name",        lambda p=props: p.name)
        check(f"GPU {i} memory (GB)", lambda p=props: f"{p.total_memory / 1e9:.1f}")

section("Distributed Backends")
import torch.distributed as dist
check("NCCL available", lambda: dist.is_nccl_available())
check("Gloo available", lambda: dist.is_gloo_available())
check("MPI  available", lambda: dist.is_mpi_available())

section("Core Packages")
for pkg in ["numpy", "matplotlib", "tqdm", "sklearn"]:
    check(pkg, lambda p=pkg: try_import(p))

section("Torchvision (for datasets & pretrained models)")
check("torchvision", lambda: try_import("torchvision"))

section("FSDP (Fully Sharded Data Parallel)")
check("torch.distributed.fsdp", lambda: try_import("torch.distributed.fsdp"))

section("Ollama Python Client")
check("ollama", lambda: try_import("ollama"))

section("Active Environment")
prefix = (
    os.environ.get("CONDA_PREFIX")
    or os.environ.get("MAMBA_ROOT_PREFIX")
    or "not detected"
)
check("env prefix", lambda: prefix)
check("GROUP_DIR set", lambda: os.environ.get("GROUP_DIR", "not set — that is OK"))

section("Quick Distributed Smoke Test (gloo / CPU)")
try:
    import subprocess, json
    result = subprocess.run(
        [sys.executable, "-c",
         "import torch.distributed as dist, os; "
         "os.environ.setdefault('MASTER_ADDR','localhost'); "
         "os.environ.setdefault('MASTER_PORT','29499'); "
         "os.environ.setdefault('RANK','0'); "
         "os.environ.setdefault('WORLD_SIZE','1'); "
         "dist.init_process_group('gloo'); "
         "print(dist.get_rank(), dist.get_world_size()); "
         "dist.destroy_process_group()"],
        capture_output=True, text=True, timeout=15,
    )
    out = result.stdout.strip()
    if out == "0 1":
        print("  [OK]   dist.init_process_group (gloo, world_size=1): passed")
    else:
        print(f"  [FAIL] dist smoke test unexpected output: {out!r} / {result.stderr.strip()!r}")
except Exception as e:
    print(f"  [FAIL] dist smoke test: {e}")

print("\n" + "=" * 55)
print("  Done.  All [OK] lines = ready for the workshop.")
print("=" * 55)
