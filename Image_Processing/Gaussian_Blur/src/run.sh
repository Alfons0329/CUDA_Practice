#!/bin/bash
set -ex

tune_streams(){
    for img in ../img_input/*jpg
    do
        fname=$(basename $img)
        fname=$(echo "${fname%%.*}")
        printf "Process %s, fname %s \n" $img $fname
        printf "Thread dim config in %dD \n" $1
        printf "Thread dim config in %dD \n" $1 >> res_$fname.csv
        for streams in 2 10 20 40 80 120 240 600 1200
        do
            sed -i "19s/#define.*/#define\ N_STREAMS\ $streams/" main_thread_$1D.cu
            mkdir -p ../report/profiling/$streams\_streams/
            make cuda -j16
            nsys profile -o ../report/profiling/$streams\_streams/nsys_data_$1D_$fname --stats true --force-overwrite true ./gb.o $img | tail -n 1 >> res_$fname.csv 2> ../report/profiling/$streams\_streams/stat_data_$1D_$fname.txt
            if [ $? -ne 0 ]
            then
                print "Program crashed with input image: %s " $img >> res_$fname.csv
            fi
            mv *jpg ../img_output/
        done
    done
}

tune_streams_mallocHost(){
    for img in ../img_input/*jpg
    do
        fname=$(basename $img)
        fname=$(echo "${fname%%.*}")
        printf "Process %s, fname %s \n" $img $fname
        printf "Thread dim config in %dD \n" $1
        printf "Thread dim config in %dD \n" $1 >> res_$fname\_mallocHost.csv
        for streams in 2 10 20 40 80 120 240 600 1200
        do
            sed -i "19s/#define.*/#define\ N_STREAMS\ $streams/" main_thread_$1D_mallocHost.cu
            mkdir -p ../report/profiling/$streams\_streams_mallocHost/
            make cuda_mallocHost -j16
            nsys profile -o ../report/profiling/$streams\_streams_mallocHost/nsys_data_$1D_$fname --stats true --force-overwrite true ./gb_mallocHost.o $img | tail -n 1 >> res_$fname\_mallocHost.csv 2> ../report/profiling/$streams\_streams_mallocHost/stat_data_$1D_$fname.txt
            if [ $? -ne 0 ]
            then
                print "Program crashed with input image: %s " $img >> res_$fname\_mallocHost.csv
            fi
            mv *jpg ../img_output/
        done
    done
}

rm -rf ../report/*
rm -f res\_*csv

tune_streams_mallocHost 2
tune_streams 2
