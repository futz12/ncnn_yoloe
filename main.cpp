// Copyright 2025 pchar.cn
// SPDX-License-Identifier: BSD-3-Clause

#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <cctype>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "SimpleTokenizer.h"
#include "MobileClip.h"
#include "YOLOE.h"

static void draw_objects(const cv::Mat &bgr, const std::vector<Object> &objects,
                         const std::vector<std::string> &labels) {
    static cv::Scalar colors[] = {
        cv::Scalar(244, 67, 54), cv::Scalar(233, 30, 99), cv::Scalar(156, 39, 176),
        cv::Scalar(103, 58, 183), cv::Scalar(63, 81, 181), cv::Scalar(33, 150, 243),
        cv::Scalar(3, 169, 244), cv::Scalar(0, 188, 212), cv::Scalar(0, 150, 136),
        cv::Scalar(76, 175, 80), cv::Scalar(139, 195, 74), cv::Scalar(205, 220, 57),
        cv::Scalar(255, 235, 59), cv::Scalar(255, 193, 7), cv::Scalar(255, 152, 0),
        cv::Scalar(255, 87, 34), cv::Scalar(121, 85, 72), cv::Scalar(158, 158, 158),
        cv::Scalar(96, 125, 139)
    };

    cv::Mat image = bgr.clone();

    for (size_t i = 0; i < objects.size(); i++) {
        const Object &obj = objects[i];
        const cv::Scalar &color = colors[i % 19];

        fprintf(stderr, "%d = %.5f at %.2f %.2f %.2f x %.2f\n",
                obj.label, obj.prob, obj.rect.x, obj.rect.y, obj.rect.width, obj.rect.height);

        for (int y = 0; y < (int) obj.rect.height; y++) {
            const uchar *maskptr = obj.mask.ptr<const uchar>(y);
            uchar *bgrptr = image.ptr<uchar>((int) obj.rect.y + y) + (int) obj.rect.x * 3;
            for (int x = 0; x < (int) obj.rect.width; x++) {
                if (maskptr[x]) {
                    bgrptr[0] = uchar(bgrptr[0] * 0.5 + color[0] * 0.5);
                    bgrptr[1] = uchar(bgrptr[1] * 0.5 + color[1] * 0.5);
                    bgrptr[2] = uchar(bgrptr[2] * 0.5 + color[2] * 0.5);
                }
                bgrptr += 3;
            }
        }

        cv::rectangle(image, obj.rect, color, 2);

        std::string name = (obj.label >= 0 && obj.label < (int) labels.size()) ? labels[obj.label] : "Object";
        char text[256];
        snprintf(text, sizeof(text), "%s %.1f%%", name.c_str(), obj.prob * 100.f);

        int baseLine = 0;
        cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

        int x = (int) obj.rect.x;
        int y = (int) obj.rect.y - label_size.height - baseLine;
        if (y < 0) y = 0;
        if (x + label_size.width > image.cols) x = image.cols - label_size.width;

        cv::rectangle(image, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)),
                      cv::Scalar(255, 255, 255), -1);
        cv::putText(image, text, cv::Point(x, y + label_size.height),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
    }

    cv::imshow("image", image);
    cv::waitKey(0);
}

static bool starts_with(const std::string &s, const std::string &prefix) {
    return s.size() >= prefix.size() && std::equal(prefix.begin(), prefix.end(), s.begin());
}

static std::string size_token(ModelSize s) {
    switch (s) {
        case ModelSize::Small:  return "11s";
        case ModelSize::Medium: return "11m";
        case ModelSize::Large:  return "11l";
    }
    return "11m";
}

static bool file_exists(const std::string& path) {
    std::ifstream f(path);
    return f.good();
}

static std::string trim(const std::string& s) {
    size_t i = 0, j = s.size();
    while (i < j && std::isspace((unsigned char)s[i])) ++i;
    while (j > i && std::isspace((unsigned char)s[j - 1])) --j;
    return s.substr(i, j - i);
}

static std::vector<std::string> split_ws(const std::string& line) {
    std::istringstream iss(line);
    std::vector<std::string> tok;
    std::string t;
    while (iss >> t) tok.push_back(t);
    return tok;
}

struct LabeledRect {
    cv::Rect rect;
    std::string tag;
};

int main()
{
    // 配置（统一用变量而非启动参数）
    const std::string model_root = "../models";
    const ModelSize model_size = ModelSize::Large;   // Small / Medium / Large
    const std::string tok = size_token(model_size);

    const int target_size = 640;
    const float prob_threshold = 0.02f;
    const float nms_threshold  = 0.45f;
    const float mask_threshold = 0.50f;

    // 推理目标图像（仅用于最终检测可视化）
    const std::string infer_img_path = "../pad.jpg";

    // 示例库根目录与类目录（满足：../example/<tag>/{images..., labels.txt}）
    const std::string example_root = "../example";
    const std::vector<std::string> example_tags = {"red_panda", "raspberry", "watch"};

    // 文本提示：一个 tag 对应多条文本
    // 你可以自行扩展每个 tag 的文本列表
    std::map<std::string, std::vector<std::string>> tag_to_texts = {
        {"person", {"person", "a person", "a human", "a man", "a woman"}},
        {"ipad", {"ipad", "tablet", "apple tablet"}},
        {"watch", {"watch"}},
    };

    // 1) 构造 YOLOE（自动加载 s/m/l 对应文件）
    YOLOE yoloe(model_size, model_root, target_size, prob_threshold, nms_threshold, mask_threshold);

    // 2) MobileCLIP 文本特征（一个 tag 多个文本）
    SimpleTokenizer tokenizer(model_root + "/bpe_simple_vocab_16e6.txt");
    MobileClip clip("mobileclip_blt");

    // 统一收集所有文本的特征，再一次性过 RepRTA
    std::vector<std::pair<std::string, cv::Mat>> text_feats_tagged; // pair<tag, 1x512>
    for (const auto& kv : tag_to_texts) {
        const std::string& tag = kv.first;
        for (const std::string& prompt_raw : kv.second) {
            std::string prompt = prompt_raw;
            std::string low = prompt;
            std::transform(low.begin(), low.end(), low.begin(), ::tolower);
            // 已经上面写了多种形式，这里不强行加 a/an
            auto tokens_list = tokenizer(prompt);
            if (tokens_list.empty()) {
                fprintf(stderr, "Tokenize failed for text: %s\n", prompt.c_str());
                continue;
            }
            cv::Mat feat = clip.encode_text(tokens_list[0]); // 1x512 float (已归一)
            text_feats_tagged.emplace_back(tag, feat);
        }
    }

    // 3) 可选：通过 RepRTA 对齐文本特征
    // 注意：RepRTA 接受一批 cv::Mat 1x512，输出 w=512,h=N；我们再拆回并按原顺序映射到 tag
    RepRTA reprta(
        model_root + "/yoloe_" + tok + "_seg/yoloe_" + tok + "_seg_reprta.ncnn.param",
        model_root + "/yoloe_" + tok + "_seg/yoloe_" + tok + "_seg_reprta.ncnn.bin"
    );
    std::vector<cv::Mat> txt_feats_raw_only;
    txt_feats_raw_only.reserve(text_feats_tagged.size());
    for (auto& p : text_feats_tagged) txt_feats_raw_only.push_back(p.second);

    ncnn::Mat txt_feats_all = reprta.forward(txt_feats_raw_only); // w=512, h=N_text

    // 4) 读取图像并获取骨干特征 P3/P4/P5（目标推理图）
    cv::Mat infer_img = cv::imread(infer_img_path, 1);
    if (infer_img.empty()) {
        fprintf(stderr, "cv::imread %s failed\n", infer_img_path.c_str());
        return -1;
    }

    // 5) 从 ../example/<tag>/labels.txt 读取图片提示（每个类别读取多个图片）
    // label 文件格式:
    //   xxx.jpg x1 y1 x2 y2 tag1  x1 y1 x2 y2 tag2  ...
    // 坐标在原图系，可能一行中包含多个对象与不同 tag
    SAVPE savpe(
        model_root + "/yoloe_" + tok + "_seg/yoloe_" + tok + "_seg_savpe.ncnn.param",
        model_root + "/yoloe_" + tok + "_seg/yoloe_" + tok + "_seg_savpe.ncnn.bin"
    );

    PromptManager pm;
    pm.set_max_per_tag(8); // 每个 tag 最多保留 8 条特征，超过则在 set_prompts 时用 PCA 压缩

    // 先把文本特征放入 pm（对应多文本->同一 tag）
    for (int i = 0; i < txt_feats_all.h; ++i) {
        const std::string& tag = text_feats_tagged[i].first;
        cv::Mat v(1, 512, CV_32F);
        std::memcpy(v.data, txt_feats_all.row(i), 512 * sizeof(float));
        pm.add_feature(tag, v);
    }

    // 遍历 example 数据构建图片提示特征
    for (const std::string& dir_tag : example_tags) {
        std::string dir_path = example_root + "/" + dir_tag;
        std::string labels_path = dir_path + "/labels.txt";
        if (!file_exists(labels_path)) {
            // 兼容 label.txt 命名
            labels_path = dir_path + "/label.txt";
            if (!file_exists(labels_path)) {
                fprintf(stderr, "warning: labels file not found for %s\n", dir_path.c_str());
                continue;
            }
        }

        std::ifstream ifs(labels_path);
        if (!ifs.is_open()) {
            fprintf(stderr, "warning: cannot open %s\n", labels_path.c_str());
            continue;
        }

        std::string line;
        while (std::getline(ifs, line)) {
            line = trim(line);
            if (line.empty() || line[0] == '#') continue;
            auto tokv = split_ws(line);
            if (tokv.size() < 6) continue;

            const std::string& filename = tokv[0];
            std::vector<LabeledRect> lr_list;

            // 解析 5 元组 (x1 y1 x2 y2 tag)
            size_t i = 1;
            while (i + 4 < tokv.size()) {
                int x1 = 0, y1 = 0, x2 = 0, y2 = 0;
                try {
                    x1 = std::stoi(tokv[i + 0]);
                    y1 = std::stoi(tokv[i + 1]);
                    x2 = std::stoi(tokv[i + 2]);
                    y2 = std::stoi(tokv[i + 3]);
                } catch (...) {
                    break;
                }
                std::string tag = tokv[i + 4];
                int w = std::max(0, x2 - x1);
                int h = std::max(0, y2 - y1);
                if (w > 0 && h > 0) {
                    lr_list.push_back({cv::Rect(x1, y1, w, h), tag});
                }
                i += 5;
            }

            if (lr_list.empty()) continue;

            // 加载该行对应图像
            std::string img_path = dir_path + "/" + filename;
            cv::Mat img = cv::imread(img_path, 1);
            if (img.empty()) {
                fprintf(stderr, "warning: cv::imread %s failed\n", img_path.c_str());
                continue;
            }

            // 提取 P3/P4/P5
            std::array<ncnn::Mat, 3> feats;
            if (yoloe.try_forward(img, feats) != 0) {
                fprintf(stderr, "warning: try_forward failed on %s\n", img_path.c_str());
                continue;
            }

            // 为该图像的所有对象生成多通道 P3 掩码
            const int P3_w = feats[0].w;
            const int P3_h = feats[0].h;
            const int Q = (int)lr_list.size();

            ncnn::Mat in_mask(P3_w, P3_h, Q);
            in_mask.fill(0.f);
            for (int q = 0; q < Q; ++q) {
                const auto& lr = lr_list[q];
                cv::Mat cv_mask = YOLOE::make_mask(img, P3_w, P3_h, lr.rect, target_size); // float 0/1
                for (int y = 0; y < P3_h; ++y) {
                    float* row = in_mask.channel(q).row(y);
                    const float* src = cv_mask.ptr<float>(y);
                    std::memcpy(row, src, P3_w * sizeof(float));
                }
            }

            // 通过 SAVPE 生成每个对象的 512 维图像原型
            ncnn::Mat img_feats = savpe.forward(feats, in_mask); // w=512, h=Q

            // 将每行特征加到对应 tag 下
            for (int q = 0; q < img_feats.h; ++q) {
                cv::Mat v(1, 512, CV_32F);
                std::memcpy(v.data, img_feats.row(q), 512 * sizeof(float));
                pm.add_feature(lr_list[q].tag, v);
            }
        }
    }

    // 6) 构建最终动态分类器（超过上限的 tag 用 PCA 压缩）
    yoloe.set_prompts(pm, /*apply_pca=*/true);

    // 7) 对目标图像做推理并绘制
    std::vector<Object> objects;
    if (yoloe.forward(infer_img, objects) != 0) {
        fprintf(stderr, "yoloe forward failed\n");
        return -1;
    }

    draw_objects(infer_img, objects, yoloe.expanded_labels());

    return 0;
}