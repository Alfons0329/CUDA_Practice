#include <bits/stdc++.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

// [BGR] in openCV, but RGB for me
unsigned char* img_read(string img_name, int& img_row, int& img_col){
    Mat img = imread(img_name);
    img_row = img.rows;
    img_col = img.cols;
    int resolution = img_row * img_col;

    unsigned char* ret = new unsigned char [3 * resolution];

    for(int i = 0; i < img_row; i++){
        for(int j = 0; j < img_col; j++){
            Vec3b color = img.at<Vec3b>(i, j);
            ret[3 * (i * img_col + j) + 0] = color[2];
            ret[3 * (i * img_col + j) + 1] = color[1];
            ret[3 * (i * img_col + j) + 2] = color[0];
        }
    }

    return ret;
}

void img_write(string img_name, unsigned char* img, const int& img_row, const int& img_col, const int thread_dim){
    Mat img_cv(img_row, img_col, CV_8UC3, Scalar(0, 0, 0));
    for (int i = 0; i < img_row; i++){
        for (int j = 0; j < img_col; j++){
            Vec3b color;
            color[2] = img[3 * (i * img_col + j) + 0];
            color[1] = img[3 * (i * img_col + j) + 1];
            color[0] = img[3 * (i * img_col + j) + 2];
            img_cv.at<Vec3b>(i, j) = color;
        }
    }

    // string name_imwrite = "./../img_output/" + p.filename() + "_" + to_string(thread_dim) + "D.jpg";
    string name_imwrite = "out_" + to_string(thread_dim) + "D.jpg";
    assert(imwrite(name_imwrite, img_cv) == true);
}
