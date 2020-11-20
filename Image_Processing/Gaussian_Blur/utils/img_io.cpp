#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>

using namespace cv;
// [BGR] in openCV, but RGB for me
unsigned char* img_read(string img_name, int& img_row, int& img_col){
    Mat img = imread(img_read);
    img_row = img.rows;
    img_col = img.cols;
    int resolution = rows * cols;

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

void img_write(string img_name, unsigned char* img, int img_row, int img_col){
    Mat img(img_row, img_col, CV_8UC3, Scalar(0, 0, 0));

    for (int i = 0; i < img_row; i++){
        for (int j = 0; j < img_col; j++){
            Vec3b color = img.at<Vec3b>(i, j);
            color[2] = ret[3 * (i * img_col + j) + 0];
            color[1] = ret[3 * (i * img_col + j) + 1];
            color[0] = ret[3 * (i * img_col + j) + 2];
        }
    }

    imwrite(img_name, img);
}