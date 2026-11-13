"""Integration validation on MATMOD against genuine saved native PyMEX outputs.

No custom eigensolver, line broadening, truncation, or spectrum normalization.
The root paths are supplied as arguments; private user paths are not published.
"""
from pathlib import Path
import argparse
import gzip
import hashlib
import json
import os
import shutil
import subprocess
import time

import h5py
import numpy as np
import yaml


def digest(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def compare(reference,candidate,scratch):
    output={}
    for filename in ('H_exciton_tb.hdf5','Eigval_exciton_tb.hdf5','Eigvec_exciton_tb.hdf5',
                     'Conductivity_exciton_tb.hdf5','H_k_tb.hdf5','Grad_H_k_tb.hdf5'):
        original=reference/filename;restored=None
        if not original.exists():
            assert original.with_suffix('.hdf5.gz').exists()
            restored=scratch/'restored-reference.hdf5'
            with gzip.open(original.with_suffix('.hdf5.gz'),'rb') as f,restored.open('wb') as g:shutil.copyfileobj(f,g)
            expected=json.loads((reference/'status.json').read_text())['file_hashes'][filename]
            assert digest(restored)==expected
            original=restored
        with h5py.File(original) as a,h5py.File(candidate/filename) as b:
            assert set(a)==set(b)
            for key in a:
                x,y=a[key],b[key];assert x.shape==y.shape
                maxdiff=0.
                for start in range(0,x.shape[0],64):
                    xx=x[start:start+64];yy=y[start:start+64]
                    maxdiff=max(maxdiff,float(np.max(abs(xx-yy))))
                output[filename+'/'+key]=maxdiff
        if restored:restored.unlink()
    x=np.loadtxt(reference/'sigma_unpolarized_tb.txt');y=np.loadtxt(candidate/'sigma_unpolarized_tb.txt')
    assert x.shape==y.shape
    output['native_TXT_max_absolute_difference']=float(np.max(abs(x-y)))
    output['native_TXT_max_relative_to_peak']=float(np.max(abs(x[:,1:]-y[:,1:]))/np.max(abs(x[:,1:])))
    # Eigenvectors/amplitudes may permute or rotate in degenerate subspaces across
    # platforms. Record their elementwise differences as diagnostics, but judge
    # equivalence using the full H, all eigenenergies and the native spectrum.
    passed=all(v<1e-9 for k,v in output.items() if k!='native_TXT_max_absolute_difference'
               and not k.startswith(('Eigvec_exciton','Conductivity_exciton')))
    return dict(passed=passed,errors=output)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project-root',type=Path,required=True)
    ap.add_argument('--scratch',type=Path,required=True);args=ap.parse_args()
    root=args.project_root;test=root/'independent-shortcut-validation';scratch=args.scratch
    scratch.mkdir(mode=0o700,parents=True,exist_ok=True);out=test/'records';out.mkdir(exist_ok=True)
    python=root/'code/.venv/bin/python';mpiexec=root/'code/.venv/bin/mpiexec'
    driver=test/'driver.py'
    driver.write_text('''import sys,time,json,os
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
''')
    initial={p.name:digest(p) for p in (root/'code/PyMEX/src').glob('*.py')}
    env=os.environ.copy();env.update(OMP_NUM_THREADS='1',OMP_THREAD_LIMIT='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1',VALIDATION_PROJECT_ROOT=str(root))
    env['PATH']=str(python.parent)+':'+env['PATH']
    report=[]

    def execute(material,mode,n,ranks,reference,baseline=False,label=''):
        name=f'{material}-{mode}-N{n:03d}-p{ranks}'+('-baseline' if baseline else '')+label
        folder=scratch/name;folder.mkdir(exist_ok=False)
        config=yaml.safe_load((reference/'pymex_tb.yaml').read_text());config['bse']['tb']['model']='model.py'
        assert config['bse']['tb']['kgrid']==[n,n,1]
        assert config['diagonalize']=={'library':'lapack'}
        (folder/'pymex_tb.yaml').write_text(yaml.safe_dump(config));shutil.copy2(reference/'model.py',folder/'model.py')
        env['VALIDATION_SOURCE']=str(root/'code/PyMEX/src' if baseline else test/'src')
        command=['nice','-n','10',str(mpiexec),'-n',str(ranks),str(python),str(driver)]
        print('RUN',name,flush=True);t=time.monotonic()
        with (folder/'run.log').open('w') as f:
            result=subprocess.run(command,cwd=folder,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=900)
        assert result.returncode==0,f'{name} failed; inspect run.log'
        check=compare(reference,folder,scratch)
        record=dict(case=name,reference=str(reference.relative_to(root)) if reference.is_relative_to(root) else reference.name,mesh=n,material=material,mode=mode,ranks=ranks,
                    unoptimized_baseline=baseline,elapsed_seconds=time.monotonic()-t,
                    timings=json.loads((folder/'timings.json').read_text()),
                    input_hashes={f:digest(folder/f) for f in ('pymex_tb.yaml','model.py')},
                    native_output_hashes={p.name:digest(p) for p in folder.iterdir() if p.suffix in ('.hdf5','.txt')},**check)
        (out/(name+'.json')).write_text(json.dumps(record,indent=2))
        shutil.copy2(folder/'run.log',out/(name+'.log'))
        assert check['passed'],record
        report.append(record);print('PASS',name,'maximum difference',max(check['errors'].values()),flush=True)
        # Small test outputs are retained in scratch; large test outputs are also
        # retained there (not in quota-limited data). Logs/hashes are durable.
        return folder

    for material in ('hBN','MoS2','MoSe2','WS2','WSe2'):
        for n in (6,9):
            reference=root/'mpi-validation-20261113/runs'/material/'independent'/f'N{n:03d}/validation-p1'
            for ranks in (1,4):execute(material,'independent',n,ranks,reference)
        for n in (30,48):
            reference=(root/f'mac-archive/optical/results/independent/runs/N{n:03d}' if material=='hBN' else
                       root/f'tmds/{material}/independent/code/tmds/optical/results/independent/runs/N{n:03d}')
            execute(material,'independent',n,1,reference)
    for material in ('hBN','MoS2'):
        for mode in ('direct','direct-exchange'):
            reference=root/'mpi-validation-20261113/runs'/material/mode/'N009/validation-p1'
            for ranks in (1,4):execute(material,mode,9,ranks,reference)
    # Fresh same-machine timing controls, followed immediately by the optimized
    # path with exactly the same inputs. Existing results are still compared above.
    for material in ('hBN','MoS2'):
        n=30
        reference=(root/'mac-archive/optical/results/independent/runs/N030' if material=='hBN' else root/f'tmds/{material}/independent/code/tmds/optical/results/independent/runs/N030')
        baseline=execute(material,'independent',n,1,reference,baseline=True)
        execute(material,'independent',n,1,baseline,label='-timed')
    unchanged=all(digest(root/'code/PyMEX/src'/name)==sha for name,sha in initial.items())
    assert unchanged
    summary=dict(actual_runs=len(report),all_passed=all(r['passed'] for r in report),
                 production_sources_unchanged=unchanged,source_hashes={p.name:digest(p) for p in (test/'src').glob('*.py')},
                 description='Only the tb independent-case matrix-row construction is optimized. Native full LAPACK diagonalization and optical routines/broadening remain unchanged. Existing interacting runs and production sources were not modified.',results=report)
    (test/'summary.json').write_text(json.dumps(summary,indent=2))
    print('ALL TESTS PASSED',len(report),flush=True)


if __name__=='__main__':main()
