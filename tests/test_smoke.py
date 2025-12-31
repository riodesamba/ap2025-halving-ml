# Makes sure the package imports (adjust later as you add code)
import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import halving_ml
def test_import():
    assert hasattr(halving_ml, "__version__")
