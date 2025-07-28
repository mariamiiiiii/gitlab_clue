#! /bin/bash

rm -rf build

cd /data/cmssw/el8_amd64_gcc12/cms/cmssw/CMSSW_15_1_0_pre4
eval `scram runtime -sh`
CUDA_BASE=$(scram tool tag cuda CUDA_BASE)
TBB_BASE=$(scram tool tag tbb TBB_BASE)
BOOST_BASE=$(scram tool tag boost BOOST_BASE)
ALPAKA_BASE=$(scram tool tag alpaka ALPAKA_BASE)
cd -


cmake \
  -DCMAKE_PREFIX_PATH="${BOOST_BASE};${TBB_BASE};${CUDA_BASE}" \
  -DCHECK_CUDA_VERSION=OFF \
  -DALPAKA_DIR="${ALPAKA_BASE}" \
  -DCMAKE_CUDA_ARCHITECTURES=89 \
  -B build \
  -S . \
  -L

cmake --build build

rm -f ../Results/results_classic*.csv

for i in {0..10}
do
  ./build/src/clue/main -i data/input/aniso_1000.csv -d 7.0 -r 10.0 -o 2 -e 10 -v -u $i
done