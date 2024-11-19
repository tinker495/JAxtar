import os
import sys
from pathlib import Path


def setup_project():
    project_root = Path(__file__).parent.parent
    if str(project_root) not in sys.path:
        sys.path.append(str(project_root))
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    return project_root
