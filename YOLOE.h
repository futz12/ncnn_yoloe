// Copyright 2025 pchar.cn
// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <string>
#include <utility>
#include <vector>
#include <map>

#include "layer.h"
#include "net.h"

struct Object {
    cv::Rect_<float> rect;
    int label; // index into expanded labels_ (class id)
    float prob;
    int gindex; // grid index for mask feature picking
    cv::Mat mask; // 8UC1 mask in box region
};

enum class ModelSize {
    Small,
    Medium,
    Large
};

// 统一提示管理器：以 tag 为键，支持文本/图片特征。
// 每个 tag 可拥有多条 512 维特征；超过上限时在 set_prompts 时可选择用 PCA 压缩。
class PromptManager {
public:
    void set_max_per_tag(int k) { max_per_tag_ = k; }
    int max_per_tag() const { return max_per_tag_; }

    // 添加一条 1x512 的特征（文本或图片），内部做 L2 归一
    void add_feature(const std::string& tag, const cv::Mat& feat_1x512);

    // 将 ncnn::Mat (w=512, h=Q) 拆分为 Q 条 1x512 并全部添加到 tag 下
    void add_features_from_ncnn_rows(const std::string& tag, const ncnn::Mat& feats_w512_hQ);

    // 生成 FUSE 所需的大矩阵 (w=512, h=N)，并返回展开后的标签序列（每条特征对应一个标签=tag）
    // 如果某 tag 下特征数 > max_per_tag，将用 PCA 压缩到 max_per_tag 条（apply_pca=true 时）
    // 注意：返回的 labels_expand_.size() == feat_fuse.h
    std::pair<ncnn::Mat, std::vector<std::string>> make_fuse_input(bool apply_pca) const;

    void clear() { tag_feats_.clear(); }

private:
    // 对一个 tag 的多个特征做 PCA 压缩为 k 条代表性特征：mean + sqrt(lambda_i) * eigenvec_i
    static std::vector<cv::Mat> pca_reduce_to_k(const std::vector<cv::Mat>& feats, int k);

private:
    // tag -> list of 1x512 float features
    std::map<std::string, std::vector<cv::Mat>> tag_feats_;
    int max_per_tag_ = 5;
};

// RepRTA 子网封装：将 text/image embedding 对齐到 YOLOE 头部空间
class RepRTA {
public:
    RepRTA(const std::string &reprta_param, const std::string &reprta_bin);
    // 输入: 1x512 float 行向量的集合；输出: ncnn::Mat(w=512, h=N)，每行一条向量
    ncnn::Mat forward(const std::vector<cv::Mat> &feat);
private:
    ncnn::Net reprta_;
};

// SAVPE 子网封装：根据 P3/P4/P5 和 P3 尺度多通道掩码，生成每个掩码的 512 维原型
class SAVPE {
public:
    SAVPE(const std::string &savpe_param, const std::string &savpe_bin);
    // feat: P3/P4/P5；vbe: ncnn::Mat(w=P3_w, h=P3_h, c=Q) 掩码
    // 返回: ncnn::Mat(w=512, h=Q)
    ncnn::Mat forward(const std::array<ncnn::Mat,3> &feat, const ncnn::Mat &vbe);
private:
    ncnn::Net savpe_;
};

class YOLOE {
public:
    // 构造函数：按 s/m/l 自动推断模型文件路径并加载 YOLOE 主干与 FUSE 子网
    // model_root: 例如 "../models"
    // 命名规则：
    //   YOLOE 主干: {root}/yoloe_{sz}_seg/yoloe_{sz}_seg.ncnn.param/bin
    //   FUSE  子网: {root}/yoloe_{sz}_seg/yoloe_{sz}_seg_fuse_head.ncnn.param/bin
    YOLOE(ModelSize size,
          const std::string &model_root,
          int target_size = 640,
          float prob_threshold = 0.10f,
          float nms_threshold = 0.45f,
          float mask_threshold = 0.5f);

    // 设置统一提示（文本/图片），内部调用 FUSE，生成并缓存动态 head 的权重/偏置
    // apply_pca: 若某 tag 特征数超过上限，将在生成输入矩阵前对其做 PCA 压缩
    void set_prompts(const PromptManager& pm, bool apply_pca);

    // 前向推理
    int forward(const cv::Mat &bgr, std::vector<Object> &objects) const;

    // 导出 P3 P4 P5 特征（供 SAVPE 使用）
    int try_forward(const cv::Mat &bgr, std::array<ncnn::Mat, 3> &feat) const;

    // 可选：获取展开后的类别标签（与动态 head 输出通道对齐）
    const std::vector<std::string> &expanded_labels() const { return labels_; }

    // 供示例：根据原图坐标生成 P3 尺度掩码（float 0/1）
    static cv::Mat make_mask(cv::Mat &img, int P3_width, int P3_heigh, cv::Rect object, const int target_size = 640);

private:
    // 内部工具
    static inline float sigmoid(float x);
    static inline float intersection_area(const Object &a, const Object &b);
    static void qsort_descent_inplace(std::vector<Object> &objects);
    static void nms_sorted_bboxes(const std::vector<Object> &objects, std::vector<int> &picked, float nms_threshold,
                                  bool agnostic = false);
    static void generate_proposals_stride(const ncnn::Mat &pred, int stride, const ncnn::Mat &in_pad,
                                          float prob_threshold, std::vector<Object> &objects);
    static void generate_proposals_pyramids(const ncnn::Mat &pred, const std::vector<int> &strides,
                                            const ncnn::Mat &in_pad, float prob_threshold,
                                            std::vector<Object> &objects);

    // 运行 FUSE 子网，把 N x 512 的输入嵌入转为若干 (weight, bias) 对
    // 返回值序列与 YOLOE 模型中注入输入 in1..in(2k) 顺序一致
    std::vector<std::pair<ncnn::Mat, ncnn::Mat> > run_fuse(const ncnn::Mat &feat_in) const;

    // 构造不同尺寸的模型路径，并配置 P3/P4/P5 的输出层名
    static void build_model_paths(ModelSize size, const std::string& root,
                                  std::string& yolo_param, std::string& yolo_bin,
                                  std::string& fuse_param, std::string& fuse_bin,
                                  std::vector<std::string>& p_layers);

private:
    // 模型
    ncnn::Net yolo_;
    ncnn::Net fuse_;

    // 推理配置
    int target_size_;
    float prob_threshold_;
    float nms_threshold_;
    float mask_threshold_;
    std::vector<int> strides_{8, 16, 32};
    int max_stride_ = 32;

    // P3/P4/P5 节点名（按尺寸配置）
    std::vector<std::string> p_layers_; // size==3, 依次 P3 P4 P5

    // set_prompts 后缓存
    std::vector<std::pair<ncnn::Mat, ncnn::Mat> > fuse_feat_; // 动态权重/偏置
    std::vector<std::string> labels_; // 展开后的标签（每条特征一个标签=tag）
};