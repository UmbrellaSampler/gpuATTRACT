#!/bin/bash -i

###Converted from AttractEasyModel by easy2model 1.1
### Generated by ATTRACT shell script generator version 0.3

set -u -e
if [ ! -f $ATTRACTDIR/../version ] ||  [ `awk '{print ($1 < 0.2)}' $ATTRACTDIR/../version` -eq 1 ]; then
  echo 'Your ATTRACT version is too old to run this protocol. Please download and install the latest version of ATTRACT'
  echo 'ATTRACT is available at: http://www.attract.ph.tum.de/services/ATTRACT/attract.tgz'
  exit 1
fi
trap "kill -- -$BASHPID; $ATTRACTDIR/shm-clean" ERR EXIT
$ATTRACTDIR/shm-clean

rm -rf result.dat result.pdb result.lrmsd result.irmsd result.fnat >& /dev/null

#name of the run
name=emATTRACT

#docking parameters
params="$ATTRACTDIR/../attract.par receptorr.pdb ligandr.pdb --fix-receptor"
paramsprep="$ATTRACTDIR/../attract.par receptorr.pdb ligandr.pdb  --ghost"
scoreparams="$ATTRACTDIR/../attract.par receptorr.pdb ligandr.pdb --score --fix-receptor"

#grid parameters
gridparams=" --grid 1 receptorgrid.grid"
if [ 1 -eq 1 ]; then ### move and change to disable parts of the protocol

echo '**************************************************************'
echo 'Reduce partner PDBs...'
echo '**************************************************************'
python $ATTRACTDIR/../allatom/aareduce.py receptor.pdb receptor-aa.pdb --chain A --pdb2pqr > receptor.mapping
python $ATTRACTDIR/../allatom/aareduce.py receptor-aa.pdb receptor-heavy.pdb --heavy --chain A > /dev/null
python $ATTRACTTOOLS/reduce.py receptor-aa.pdb receptorr.pdb --chain A > /dev/null
python $ATTRACTDIR/../allatom/aareduce.py ligand.pdb ligand-aa.pdb --chain B --pdb2pqr > ligand.mapping
python $ATTRACTDIR/../allatom/aareduce.py ligand-aa.pdb ligand-heavy.pdb --heavy --chain B > /dev/null
python $ATTRACTTOOLS/reduce.py ligand-aa.pdb ligandr.pdb --chain B > /dev/null

echo '**************************************************************'
echo 'Reduce reference PDBs...'
echo '**************************************************************'
python $ATTRACTDIR/../allatom/aareduce.py receptor-rmsd.pdb refe-rmsd-1.pdb --heavy --pdb2pqr > /dev/null
python $ATTRACTDIR/../allatom/aareduce.py ligand-rmsd.pdb refe-rmsd-2.pdb --heavy --pdb2pqr > /dev/null

echo '**************************************************************'
echo 'Generate starting structures...'
echo '**************************************************************'
cat $ATTRACTDIR/../rotation.dat > rotation.dat
$ATTRACTDIR/translate receptorr.pdb ligandr.pdb > translate.dat
$ATTRACTDIR/systsearch > systsearch.dat


echo '**************************************************************'
echo 'calculate receptorgrid grid'
echo '**************************************************************'
awk '{print substr($0,58,2)}' ligandr.pdb | sort -nu > receptorgrid.alphabet
$ATTRACTDIR/make-grid-omp receptorr.pdb $ATTRACTDIR/../attract.par 5.0 7.0 receptorgrid.grid  --alphabet receptorgrid.alphabet

fi

start=systsearch.dat

if [ 1 -eq 1 ]; then


function em () {

$ASDIR/emATTRACT --dof systsearch.dat -p $ATTRACTDIR/../attract.par -a receptorgrid.alphabet -d 0  > out_$name.dat

}

echo '**************************************************************'
echo 'Docking'
echo '**************************************************************'


time (em) 2> $name.docking.time  ## timing does not work. Why?



echo '**************************************************************'
echo 'Final rescoring'
echo '**************************************************************'
#$ATTRACTDIR/attract out_$name.dat $scoreparams --rcut 50.0 > out_$name.score 
time (python $ATTRACTDIR/../protocols/attract.py out_$name.dat $scoreparams --rcut 50.0 --output out_$name.score --np 8 --chunks 8 ) 2> $name.re-scoring.8t.time
fi

if [ 1 -eq 1 ]; then     
echo '**************************************************************'
echo 'Merge the scores with the structures'
echo '**************************************************************'
python $ATTRACTTOOLS/fill-energies.py out_$name.dat out_$name.score > out_$name-scored.dat

echo '**************************************************************'
echo 'Sort structures'
echo '**************************************************************'
python $ATTRACTTOOLS/sort.py out_$name-scored.dat > out_$name-sorted.dat

echo '**************************************************************'
echo 'Remove redundant structures'
echo '**************************************************************'
$ATTRACTDIR/deredundant out_$name-sorted.dat 2 | python $ATTRACTTOOLS/fill-deredundant.py /dev/stdin out_$name-sorted.dat > out_$name-sorted-dr.dat

echo '**************************************************************'
echo 'Soft-link the final results'
echo '**************************************************************'
ln -s out_$name-sorted-dr.dat result.dat

echo '**************************************************************'
echo 'collect top 50 structures:'
echo '**************************************************************'
$ATTRACTTOOLS/top out_$name-sorted-dr.dat 50 > out_$name-top50.dat
$ATTRACTDIR/collect out_$name-top50.dat receptor-aa.pdb ligand-aa.pdb > out_$name-top50.pdb
ln -s out_$name-top50.pdb result.pdb

echo '**************************************************************'
echo 'calculate backbone ligand RMSD'
echo '**************************************************************'
python $ATTRACTDIR/lrmsd.py out_$name-sorted-dr.dat ligand-aa.pdb ligand-rmsd.pdb --receptor receptor-aa.pdb > out_$name-sorted-dr.lrmsd
ln -s out_$name-sorted-dr.lrmsd result.lrmsd

echo '**************************************************************'
echo 'calculate backbone interface RMSD'
echo '**************************************************************'
python $ATTRACTDIR/irmsd.py out_$name-sorted-dr.dat receptor-heavy.pdb refe-rmsd-1.pdb ligand-heavy.pdb refe-rmsd-2.pdb  > out_$name-sorted-dr.irmsd
ln -s out_$name-sorted-dr.irmsd result.irmsd

echo '**************************************************************'
echo 'calculate fraction of native contacts'
echo '**************************************************************'
python $ATTRACTDIR/fnat.py out_$name-sorted-dr.dat 5 receptor-heavy.pdb refe-rmsd-1.pdb ligand-heavy.pdb refe-rmsd-2.pdb > out_$name-sorted-dr.fnat
ln -s out_$name-sorted-dr.fnat result.fnat

fi ### move to disable parts of the protocol
