#!/bin/bash
set -x

tune_streams(){
    for img in ../img_input/*jpg
    do
        fname=$(basename $img)
        fname=$(echo "${fname%%.*}")
        printf "Process %s, fname %s \n" $img $fname
        printf "Thread dim config in %dD \n" $1
        printf "Thread dim config in %dD \n" $1 >> res_$fname.csv
        for streams in 2 4 8 10 20 40 80 120 240
        do
            sed -i "18s/#define.*/#define\ N_STREAMS\ $streams/" main_thread_$1D.cu
            mkdir -p ../report/profiling/$streams\_streams/
            make cuda_$1D -j16
            nsys profile -o ../report/profiling/$streams\_streams/nsys_data_$1D_$fname --stats true --force-overwrite true ./src/gb_$1D.o $img 2>  ../report/profiling/$streams\_streams/stat_data_$1D_$fname.txt
            ./gb_$1D.o $img | tail -n 1 >> res_$fname.csv
            if [ $? -ne 0 ]
            then
                print "Program crashed with input image: %s " $img >> res_$fname.csv
            fi
            mv *jpg ../img_output/
        done
    done
}

rm -f ./res_*csv
tune_streams 1
tune_streams 2
