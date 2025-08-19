//
// Created by ice on 25-7-21.
//

#include "MobileClip.h"

MobileClip::MobileClip(const string &model_name) {
    string model_path = "../models/" + model_name + "_export/";

    image_encoder.load_param((model_path + "image_encoder.ncnn.param").c_str());
    image_encoder.load_model((model_path + "image_encoder.ncnn.bin").c_str());

    text_encoder.load_param((model_path + "text_encoder.ncnn.param").c_str());
    text_encoder.load_model((model_path + "text_encoder.ncnn.bin").c_str());

    projection.load_param((model_path + "projection_layer.ncnn.param").c_str());
    projection.load_model((model_path + "projection_layer.ncnn.bin").c_str());

    m_model_name = model_name;
}

cv::Mat MobileClip::encode_image(cv::Mat image) {
    const int target_size = (m_model_name == "mobileclip_b" || m_model_name == "mobileclip_blt") ? 224 : 256;

    if (!image.isContinuous()) {
        image = image.clone();
    }

    ncnn::Mat in = ncnn::Mat::from_pixels_resize(image.data, ncnn::Mat::PIXEL_BGR2RGB, image.cols, image.rows,
                                                 target_size, target_size);

    float mean_vals[3] = {0, 0, 0};
    float norm_vals[3] = {1.0f / 255, 1.0f / 255, 1.0f / 255};
    in.substract_mean_normalize(mean_vals, norm_vals);

    auto ex = image_encoder.create_extractor();
    ex.input("in0", in);

    ncnn::Mat out0;
    ex.extract("out0", out0);

    cv::Mat result(out0.h, out0.w, CV_32F);
    memcpy(result.data, out0.data, out0.h * out0.w * sizeof(float));

    cv::normalize(result, result);
    return result;
}

cv::Mat MobileClip::encode_text(vector<int> tokens) {
    const int max_length = 77;
    ncnn::Mat in(max_length);
    in.fill(0.0f);
    for (int i = 0; i < tokens.size() && i < max_length; i++) {
        in.row<int>(0)[i] = tokens[i];
    }
    ncnn::Mat out;
    auto ex = text_encoder.create_extractor();
    ex.input("in0", in);
    ex.extract("out0", out);

    int eot_token_index = 0;
    for (int i = 0; i < tokens.size(); i++) {
        if (tokens[i] == 49407) { // find eot
            eot_token_index = i;
            break;
        }
    }
    ncnn::Mat text_embed = out.row_range(eot_token_index, 1);

    auto ex2 = projection.create_extractor();
    ex2.input("in0", text_embed);
    ncnn::Mat out2;
    ex2.extract("out0", out2);

    cv::Mat result(1, out.w, CV_32F);
    memcpy(result.data, out2.data, out2.w * sizeof(float));
    cv::normalize(result, result);
    return result;
}
