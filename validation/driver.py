import sys,time,json,os
from pathlib import Path
from mpi4py import MPI
root=Path(os.environ['VALIDATION_PROJECT_ROOT'])
sys.path[:0]=[os.environ['VALIDATION_SOURCE'],str(root/'code/.build/pymex')]
from bse import BSE
c=MPI.COMM_WORLD;times={};c.Barrier();t=time.monotonic();b=BSE('pymex_tb.yaml');c.Barrier()
times['setup_s']=time.monotonic()-t
for name,fn in [('matrix',b.write_exciton_H),('diagonalization',b.diagon_BSE),('optical',b.optical_conductivity)]:
    t=time.monotonic();fn();c.Barrier();times[name+'_s']=time.monotonic()-t
if c.rank==0:Path('timings.json').write_text(json.dumps(times,indent=2))
