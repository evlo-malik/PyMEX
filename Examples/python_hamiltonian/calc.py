"""Run PyMEX in the current example directory; set PYTHONPATH to its source/build."""
from bse import BSE


if __name__ == "__main__":
    calculation = BSE("pymex_tb.yaml")
    calculation.write_exciton_H()
    calculation.diagon_BSE()
    calculation.optical_conductivity()
