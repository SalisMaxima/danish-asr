"""Invoke tasks for Danish ASR.

Tasks are organized into namespaces for better organization.

Usage:
    invoke <namespace>.<task> [options]

Examples:
    invoke core.setup-dev
    invoke train.train
    invoke quality.ci
"""

import importlib.util
from pathlib import Path

from invoke import Collection


def load_module_from_file(module_name, file_path):
    """Load a module from a file path."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


tasks_dir = Path(__file__).parent / "tasks"

core = load_module_from_file("core", tasks_dir / "core.py")
data = load_module_from_file("data", tasks_dir / "data.py")
train = load_module_from_file("train", tasks_dir / "train.py")
quality = load_module_from_file("quality", tasks_dir / "quality.py")
docs = load_module_from_file("docs", tasks_dir / "docs.py")
utils = load_module_from_file("utils", tasks_dir / "utils.py")

namespace = Collection()

namespace.add_collection(Collection.from_module(core), name="core")
namespace.add_collection(Collection.from_module(data), name="data")
namespace.add_collection(Collection.from_module(train), name="train")
namespace.add_collection(Collection.from_module(quality), name="quality")
namespace.add_collection(Collection.from_module(docs), name="docs")
namespace.add_collection(Collection.from_module(utils), name="utils")
