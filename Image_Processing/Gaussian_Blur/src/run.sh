#!/bin/bash
set -xe
> res.csv

tune_streams(){
    printf "Thread dim config in %dD \n" $1
    printf "Thread dim config in %dD \n" $1 >> res.csv
    for streams in 2 4 8 10 20
    do
        sed -i "18s/#define.*/#define\ N_STREAMS\ $streams/" main_thread_$1D.cu
        mkdir -p ../report/profiling/$streams\_streams/
        make cuda_$1D -j16
        nsys profile -o ../report/profiling/$streams\_streams/profile_data --stats true ./src/gb_$1D.o ../$2 2>  ../report/profiling/$streams\_streams/profile_data.txt
        ./gb_$1D.o $2 | tail -n 1 >> res.csv
        mv *jpg ../img_output/
    done
}

if [ $# -ne 1 ]
then
    printf "Usage: ./run.sh <imge_file_path>"
    exit -1
fi

tune_streams 1 $1
tune_streams 2 $1

