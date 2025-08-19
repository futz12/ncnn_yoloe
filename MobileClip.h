//
// Created by ice on 25-7-21.
//

#ifndef MOBILECLIP_H
#define MOBILECLIP_H

#include <iostream>
#include <string>

#include <ncnn/net.h>
#include <ncnn/mat.h>

#include <opencv2/opencv.hpp>

using namespace std;


class MobileClip {
public:
    // mobileclip_b.pt
    // mobileclip_s0.pt
    // mobileclip_s1.pt
    // mobileclip_s2.pt
    explicit MobileClip(const string &model_name);

    cv::Mat encode_image(cv::Mat image);

    cv::Mat encode_text(vector<int> tokens);
private:
    ncnn::Net image_encoder;
    ncnn::Net text_encoder;
    ncnn::Net projection;

    std::string m_model_name;
};


#endif //MOBILECLIP_H
