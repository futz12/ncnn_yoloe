// Copyright 2025 pchar.cn
// SPDX-License-Identifier: BSD-3-Clause

#include "YOLOE.h"

#include <algorithm>
#include <cfloat>
#include <cstdio>
#include <cstring>
#include <numeric>

const char *Large_P_Layer[] = {
    "509", "615", "713"
}; // P3 P4 P5 P6 P7

const char *Medium_P_Layer[] = {
        "300", "376", "444"
}; // P3 P4 P5

const char *Small_P_Layer[] = {
    "224", "281", "349"
}; // P3 P4 P5

/**********************
 * PromptManager 实现
 **********************/
static inline void l2_normalize_1x512(cv::Mat& v)
{
    CV_Assert(v.total() == 512 && v.type() == CV_32F);
    float* p = v.ptr<float>(0);
    double ss = 0.0;
    for (int i = 0; i < 512; ++i) ss += double(p[i]) * double(p[i]);
    ss = std::sqrt(std::max(1e-12, ss));
    float inv = float(1.0 / ss);
    for (int i = 0; i < 512; ++i) p[i] *= inv;
}

void PromptManager::add_feature(const std::string& tag, const cv::Mat& feat_1x512)
{
    CV_Assert(feat_1x512.rows == 1 && feat_1x512.cols == 512 && feat_1x512.type() == CV_32F);
    cv::Mat v = feat_1x512.clone();
    l2_normalize_1x512(v);
    tag_feats_[tag].push_back(std::move(v));
}

void PromptManager::add_features_from_ncnn_rows(const std::string& tag, const ncnn::Mat& feats_w512_hQ)
{
    CV_Assert(feats_w512_hQ.w == 512);
    for (int i = 0; i < feats_w512_hQ.h; ++i)
    {
        cv::Mat v(1, 512, CV_32F);
        std::memcpy(v.data, feats_w512_hQ.row(i), 512 * sizeof(float));
        l2_normalize_1x512(v);
        tag_feats_[tag].push_back(std::move(v));
    }
}

std::vector<cv::Mat> PromptManager::pca_reduce_to_k(const std::vector<cv::Mat>& feats, int k)
{
    int N = (int)feats.size();
    if (N <= k) return feats;

    cv::Mat data(N, 512, CV_32F);
    for (int i = 0; i < N; ++i)
        std::memcpy(data.ptr<float>(i), feats[i].ptr<float>(0), 512 * sizeof(float));

    cv::PCA pca(data, cv::Mat(), cv::PCA::DATA_AS_ROW, k); // 保留 k 主成分

    std::vector<cv::Mat> out;
    out.reserve(k);
    for (int i = 0; i < k; ++i)
    {
        cv::Mat comp = pca.mean.clone(); // 1x512
        float scale = std::sqrt(std::max(0.f, pca.eigenvalues.at<float>(0, i)));
        for (int j = 0; j < 512; ++j)
        {
            comp.at<float>(0, j) += scale * pca.eigenvectors.at<float>(i, j);
        }
        l2_normalize_1x512(comp);
        out.push_back(std::move(comp));
    }
    return out;
}

std::pair<ncnn::Mat, std::vector<std::string>> PromptManager::make_fuse_input(bool apply_pca) const
{
    int total_after = 0;
    for (auto& kv : tag_feats_)
        total_after += std::min(max_per_tag_, (int)kv.second.size());

    ncnn::Mat fuse_in(512, total_after);
    std::vector<std::string> labels_expand;
    labels_expand.reserve(total_after);

    int row = 0;
    for (auto& kv : tag_feats_)
    {
        const std::string& tag = kv.first;
        const std::vector<cv::Mat>& feats = kv.second;

        std::vector<cv::Mat> use_feats;
        if ((int)feats.size() <= max_per_tag_)
        {
            use_feats = feats;
        }
        else
        {
            if (apply_pca) use_feats = pca_reduce_to_k(feats, max_per_tag_);
            else use_feats.assign(feats.begin(), feats.begin() + max_per_tag_);
        }

        for (const auto& v : use_feats)
        {
            std::memcpy(fuse_in.row(row), v.ptr<float>(0), 512 * sizeof(float));
            labels_expand.push_back(tag);
            ++row;
        }
    }
    return {fuse_in, labels_expand};
}

/**********************
 * RepRTA / SAVPE 实现
 **********************/
RepRTA::RepRTA(const std::string &reprta_param, const std::string &reprta_bin)
{
    reprta_.load_param(reprta_param.c_str());
    reprta_.load_model(reprta_bin.c_str());
}

ncnn::Mat RepRTA::forward(const std::vector<cv::Mat> &feat)
{
    ncnn::Mat feat_in(512, (int) feat.size());
    for (size_t i = 0; i < feat.size(); i++) {
        std::memcpy(feat_in.row((int) i), feat[i].data, feat[i].total() * sizeof(float));
    }
    auto ex = reprta_.create_extractor();
    ex.input("in0", feat_in);
    ncnn::Mat out;
    if (ex.extract("out0", out) != 0) {
        fprintf(stderr, "RepRTA: extract output failed\n");
        return ncnn::Mat();
    }
    return out; // w=512, h=N
}

SAVPE::SAVPE(const std::string &savpe_param, const std::string &savpe_bin)
{
    savpe_.load_param(savpe_param.c_str());
    savpe_.load_model(savpe_bin.c_str());
}

ncnn::Mat SAVPE::forward(const std::array<ncnn::Mat,3> &feat, const ncnn::Mat &vbe)
{
    ncnn::Extractor ex = savpe_.create_extractor();
    ex.input("in0", feat[0]);
    ex.input("in1", feat[1]);
    ex.input("in2", feat[2]);
    ex.input("in3", vbe);

    ncnn::Mat out;
    if (ex.extract("out0", out) != 0) {
        fprintf(stderr, "SAVPE: extract output failed\n");
        return ncnn::Mat();
    }
    return out; // 预期 w=512, h=Q
}

/**********************
 * YOLOE 实现
 **********************/
static std::string size_to_token(ModelSize sz)
{
    switch (sz) {
        case ModelSize::Small:  return "11s";
        case ModelSize::Medium: return "11m";
        case ModelSize::Large:  return "11l";
    }
    return "11m";
}

void YOLOE::build_model_paths(ModelSize size, const std::string& root,
                              std::string& yolo_param, std::string& yolo_bin,
                              std::string& fuse_param, std::string& fuse_bin,
                              std::vector<std::string>& p_layers)
{
    std::string tok = size_to_token(size);
    std::string dir = root + "/yoloe_" + tok + "_seg/";

    yolo_param = dir + "yoloe_" + tok + "_seg.ncnn.param";
    yolo_bin   = dir + "yoloe_" + tok + "_seg.ncnn.bin";
    fuse_param = dir + "yoloe_" + tok + "_seg_fuse_head.ncnn.param";
    fuse_bin   = dir + "yoloe_" + tok + "_seg_fuse_head.ncnn.bin";

    p_layers.clear();
    if (size == ModelSize::Medium)
    {
        p_layers = { Medium_P_Layer[0], Medium_P_Layer[1], Medium_P_Layer[2] };
    }
    else if (size == ModelSize::Large)
    {
        p_layers = { Large_P_Layer[0], Large_P_Layer[1], Large_P_Layer[2] };
    }
    else // Small
    {
        p_layers = { Small_P_Layer[0], Small_P_Layer[1], Small_P_Layer[2] };
    }
}

YOLOE::YOLOE(ModelSize size,
             const std::string& model_root,
             int target_size,
             float prob_threshold,
             float nms_threshold,
             float mask_threshold)
    : target_size_(target_size),
      prob_threshold_(prob_threshold),
      nms_threshold_(nms_threshold),
      mask_threshold_(mask_threshold)
{
    std::string yolo_param, yolo_bin, fuse_param, fuse_bin;
    build_model_paths(size, model_root, yolo_param, yolo_bin, fuse_param, fuse_bin, p_layers_);

    // YOLOE 主干
    yolo_.opt.use_vulkan_compute = false;
    yolo_.opt.use_packing_layout = false;

    if (yolo_.load_param(yolo_param.c_str()) != 0 || yolo_.load_model(yolo_bin.c_str()) != 0)
    {
        fprintf(stderr, "Failed to load yoloe model: %s / %s\n", yolo_param.c_str(), yolo_bin.c_str());
    }

    // FUSE 子网（不再注册 ones_like，自定义算子已不需要）
    fuse_.opt.use_vulkan_compute = false;
    fuse_.opt.use_packing_layout = false;

    if (fuse_.load_param(fuse_param.c_str()) != 0 || fuse_.load_model(fuse_bin.c_str()) != 0)
    {
        fprintf(stderr, "Failed to load fuse model: %s / %s\n", fuse_param.c_str(), fuse_bin.c_str());
    }
}

void YOLOE::set_prompts(const PromptManager& pm, bool apply_pca)
{
    auto packed = pm.make_fuse_input(apply_pca);
    const ncnn::Mat& fuse_in = packed.first;
    const std::vector<std::string>& labels_expand = packed.second;

    if (fuse_in.empty() || (int)labels_expand.size() != fuse_in.h)
    {
        fprintf(stderr, "set_prompts: invalid packed inputs, feature count must match labels\n");
        fuse_feat_.clear();
        labels_.clear();
        return;
    }

    // 运行 FUSE，缓存动态权重
    fuse_feat_ = run_fuse(fuse_in);
    labels_ = labels_expand;
}

int YOLOE::forward(const cv::Mat& bgr, std::vector<Object>& objects) const
{
    if (bgr.empty())
        return -1;
    if (fuse_feat_.empty())
    {
        fprintf(stderr, "forward: fuse features not set. Call set_prompts() first.\n");
        return -1;
    }

    const int img_w = bgr.cols;
    const int img_h = bgr.rows;

    // 计算 letterbox 尺寸
    int w = img_w;
    int h = img_h;
    float scale = 1.f;
    if (w > h)
    {
        scale = (float)target_size_ / w;
        w = target_size_;
        h = int(h * scale);
    }
    else
    {
        scale = (float)target_size_ / h;
        h = target_size_;
        w = int(w * scale);
    }

    // 预处理：resize + pad + normalize
    ncnn::Mat in = ncnn::Mat::from_pixels_resize(bgr.data, ncnn::Mat::PIXEL_BGR2RGB, img_w, img_h, w, h);

    int wpad = (w + max_stride_ - 1) / max_stride_ * max_stride_ - w;
    int hpad = (h + max_stride_ - 1) / max_stride_ * max_stride_ - h;

    ncnn::Mat in_pad;
    ncnn::copy_make_border(in, in_pad, hpad / 2, hpad - hpad / 2, wpad / 2, wpad - wpad / 2, ncnn::BORDER_CONSTANT, 114.f);

    const float norm_vals[3] = {1 / 255.f, 1 / 255.f, 1 / 255.f};
    in_pad.substract_mean_normalize(0, norm_vals);

    // 推理
    ncnn::Extractor ex = yolo_.create_extractor();
    ex.input("in0", in_pad);

    // 注入动态 weight/bias
    for (size_t i = 0; i < fuse_feat_.size(); i++)
    {
        const auto& wb = fuse_feat_[i];
        std::string weight_name = "in" + std::to_string(i * 2 + 1);
        std::string bias_name   = "in" + std::to_string(i * 2 + 2);
        ex.input(weight_name.c_str(), wb.first);
        ex.input(bias_name.c_str(),  wb.second);
    }

    // 输出
    ncnn::Mat out0; // det head
    ncnn::Mat out1; // mask feats (per grid)
    ncnn::Mat out2; // mask protos
    if (ex.extract("out0", out0) != 0 || ex.extract("out1", out1) != 0 || ex.extract("out2", out2) != 0)
    {
        fprintf(stderr, "YOLOE: extract outputs failed\n");
        return -1;
    }

    // 生成 proposals
    std::vector<Object> proposals;
    generate_proposals_pyramids(out0, strides_, in_pad, prob_threshold_, proposals);
    qsort_descent_inplace(proposals);

    std::vector<int> picked;
    nms_sorted_bboxes(proposals, picked, nms_threshold_);

    const int count = (int)picked.size();
    if (count == 0)
    {
        objects.clear();
        return 0;
    }

    // 构建 objects + 选取 mask_feat
    ncnn::Mat objects_mask_feat(out1.w, 1, count); // [count, K], K=out1.w
    objects.resize(count);

    for (int i = 0; i < count; i++)
    {
        objects[i] = proposals[picked[i]];

        // 映射回原图坐标
        float x0 = (objects[i].rect.x - (wpad / 2)) / scale;
        float y0 = (objects[i].rect.y - (hpad / 2)) / scale;
        float x1 = (objects[i].rect.x + objects[i].rect.width  - (wpad / 2)) / scale;
        float y1 = (objects[i].rect.y + objects[i].rect.height - (hpad / 2)) / scale;

        x0 = std::max(std::min(x0, (float)(img_w - 1)), 0.f);
        y0 = std::max(std::min(y0, (float)(img_h - 1)), 0.f);
        x1 = std::max(std::min(x1, (float)(img_w - 1)), 0.f);
        y1 = std::max(std::min(y1, (float)(img_h - 1)), 0.f);

        objects[i].rect.x = x0;
        objects[i].rect.y = y0;
        objects[i].rect.width  = x1 - x0;
        objects[i].rect.height = y1 - y0;

        // 取该 grid 的 mask feature
        std::memcpy(objects_mask_feat.channel(i), out1.row(objects[i].gindex), out1.w * sizeof(float));
    }

    // 计算所有对象的 masks: GEMM(objects_mask_feat, mask_protos)
    ncnn::Mat objects_mask;
    {
        ncnn::Layer* gemm = ncnn::create_layer("Gemm");

        ncnn::ParamDict pd;
        pd.set(6, 1);                             // constantC
        pd.set(7, count);                         // constantM
        pd.set(8, out2.w * out2.h);               // constantN
        pd.set(9, out1.w);                        // constantK
        pd.set(10, -1);                           // constant_broadcast_type_C
        pd.set(11, 1);                            // output_N1M
        gemm->load_param(pd);

        ncnn::Option opt;
        opt.num_threads = 1;
        opt.use_packing_layout = false;

        gemm->create_pipeline(opt);

        std::vector<ncnn::Mat> gemm_inputs(2);
        gemm_inputs[0] = objects_mask_feat;                                // [M, K]
        gemm_inputs[1] = out2.reshape(out2.w * out2.h, 1, out2.c);         // [K, N]
        std::vector<ncnn::Mat> gemm_outputs(1);
        gemm->forward(gemm_inputs, gemm_outputs, opt);
        objects_mask = gemm_outputs[0].reshape(out2.w, out2.h, count);     // [N, M] -> [W, H, M]

        gemm->destroy_pipeline(opt);
        delete gemm;
    }
    {
        ncnn::Layer* sig = ncnn::create_layer("Sigmoid");
        ncnn::Option opt;
        opt.num_threads = 1;
        opt.use_packing_layout = false;
        sig->create_pipeline(opt);
        sig->forward_inplace(objects_mask, opt);
        sig->destroy_pipeline(opt);
        delete sig;
    }

    // resize mask 到原图尺度
    {
        ncnn::Mat objects_mask_resized;
        ncnn::resize_bilinear(objects_mask, objects_mask_resized, (int)(in_pad.w / scale), (int)(in_pad.h / scale));
        objects_mask = objects_mask_resized;
    }

    // 每个对象裁剪出其框内的二值 mask
    for (int i = 0; i < count; i++)
    {
        Object& obj = objects[i];

        const ncnn::Mat mm = objects_mask.channel(i);
        obj.mask = cv::Mat((int)obj.rect.height, (int)obj.rect.width, CV_8UC1);

        for (int y = 0; y < (int)obj.rect.height; y++)
        {
            const float* pmm = mm.row((int)(obj.rect.y + y)) + (int)(obj.rect.x);
            uchar* pmask = obj.mask.ptr<uchar>(y);
            for (int x = 0; x < (int)obj.rect.width; x++)
            {
                pmask[x] = pmm[x] > mask_threshold_ ? 1 : 0;
            }
        }
    }

    return 0;
}

int YOLOE::try_forward(const cv::Mat &bgr, std::array<ncnn::Mat, 3> &feat) const
{
    if (bgr.empty())
        return -1;

    // 检查节点是否已配置
    if (p_layers_.size() != 3 || p_layers_[0].empty() || p_layers_[1].empty() || p_layers_[2].empty())
    {
        fprintf(stderr, "try_forward: P3/P4/P5 layer names not configured for this model size.\n");
        return -2;
    }

    const int img_w = bgr.cols;
    const int img_h = bgr.rows;

    // 计算 letterbox 尺寸
    int w = img_w;
    int h = img_h;
    float scale = 1.f;
    if (w > h) {
        scale = (float) target_size_ / w;
        w = target_size_;
        h = int(h * scale);
    } else {
        scale = (float) target_size_ / h;
        h = target_size_;
        w = int(w * scale);
    }

    // 预处理：resize + pad + normalize
    ncnn::Mat in = ncnn::Mat::from_pixels_resize(bgr.data, ncnn::Mat::PIXEL_BGR2RGB, img_w, img_h, w, h);

    int wpad = (w + max_stride_ - 1) / max_stride_ * max_stride_ - w;
    int hpad = (h + max_stride_ - 1) / max_stride_ * max_stride_ - h;

    ncnn::Mat in_pad;
    ncnn::copy_make_border(in, in_pad, hpad / 2, hpad - hpad / 2, wpad / 2, wpad - wpad / 2, ncnn::BORDER_CONSTANT,
                           114.f);

    const float norm_vals[3] = {1 / 255.f, 1 / 255.f, 1 / 255.f};
    in_pad.substract_mean_normalize(0, norm_vals);

    // 推理
    ncnn::Extractor ex = yolo_.create_extractor();
    ex.input("in0", in_pad);

    ncnn::Mat P3; // P3
    ncnn::Mat P4; // P4
    ncnn::Mat P5; // P5

    if (ex.extract(p_layers_[0].c_str(), P3) != 0 ||
        ex.extract(p_layers_[1].c_str(), P4) != 0 ||
        ex.extract(p_layers_[2].c_str(), P5) != 0)
    {
        fprintf(stderr, "try_forward: extract P3/P4/P5 failed\n");
        return -3;
    }

    feat[0] = P3;
    feat[1] = P4;
    feat[2] = P5;

    return 0;
}

std::vector<std::pair<ncnn::Mat, ncnn::Mat>> YOLOE::run_fuse(const ncnn::Mat& feat_in) const
{
    ncnn::Extractor ex = fuse_.create_extractor();
    ex.input("in0", feat_in);

    // 输出对数 = 输出名数量 / 2，约定 out0=weight0, out1=bias0, out2=weight1, out3=bias1, ...
    size_t output_size = fuse_.output_names().size() / 2;

    std::vector<std::pair<ncnn::Mat, ncnn::Mat>> ret(output_size);

    for (size_t i = 0; i < output_size; i++)
    {
        ncnn::Mat out_weight, out_bias;
        ex.extract(("out" + std::to_string(i)).c_str(), out_weight);
        ex.extract(("out" + std::to_string(i + output_size)).c_str(), out_bias);

        // 重塑为卷积期望的形状（与 YOLOE 动态输入保持一致）
        out_weight = out_weight.reshape(1, 1, out_weight.w, out_weight.h);
        ret[i] = std::make_pair(out_weight, out_bias);
    }

    return ret;
}

inline float YOLOE::sigmoid(float x)
{
    return 1.f / (1.f + expf(-x));
}

inline float YOLOE::intersection_area(const Object& a, const Object& b)
{
    cv::Rect_<float> inter = a.rect & b.rect;
    return inter.area();
}

static void qsort_descent_inplace_impl(std::vector<Object>& objects, int left, int right)
{
    int i = left;
    int j = right;
    float p = objects[(left + right) / 2].prob;

    while (i <= j)
    {
        while (objects[i].prob > p) i++;
        while (objects[j].prob < p) j--;
        if (i <= j)
        {
            std::swap(objects[i], objects[j]);
            i++; j--;
        }
    }
    if (left < j) qsort_descent_inplace_impl(objects, left, j);
    if (i < right) qsort_descent_inplace_impl(objects, i, right);
}

void YOLOE::qsort_descent_inplace(std::vector<Object>& objects)
{
    if (objects.empty()) return;
    qsort_descent_inplace_impl(objects, 0, (int)objects.size() - 1);
}

void YOLOE::nms_sorted_bboxes(const std::vector<Object>& objects, std::vector<int>& picked, float nms_threshold, bool agnostic)
{
    picked.clear();
    const int n = (int)objects.size();
    std::vector<float> areas(n);
    for (int i = 0; i < n; i++) areas[i] = objects[i].rect.area();

    for (int i = 0; i < n; i++)
    {
        const Object& a = objects[i];

        int keep = 1;
        for (int j = 0; j < (int)picked.size(); j++)
        {
            const Object& b = objects[picked[j]];
            if (!agnostic && a.label != b.label) continue;

            float inter_area = intersection_area(a, b);
            float union_area = areas[i] + areas[picked[j]] - inter_area;
            if (inter_area / union_area > nms_threshold) { keep = 0; break; }
        }
        if (keep) picked.push_back(i);
    }
}

void YOLOE::generate_proposals_stride(const ncnn::Mat& pred, int stride, const ncnn::Mat& in_pad, float prob_threshold, std::vector<Object>& objects)
{
    const int w = in_pad.w;
    const int h = in_pad.h;

    const int num_grid_x = w / stride;
    const int num_grid_y = h / stride;

    const int reg_max_1 = 16;
    const int num_class = pred.w - reg_max_1 * 4;

    for (int gy = 0; gy < num_grid_y; gy++)
    {
        for (int gx = 0; gx < num_grid_x; gx++)
        {
            const ncnn::Mat pred_grid = pred.row_range(gy * num_grid_x + gx, 1);

            // 分类分数
            int label = -1;
            float score = -FLT_MAX;
            {
                const ncnn::Mat pred_score = pred_grid.range(reg_max_1 * 4, num_class);
                for (int k = 0; k < num_class; k++)
                {
                    float s = pred_score[k];
                    if (s > score) { label = k; score = s; }
                }
                score = sigmoid(score);
            }

            if (score >= prob_threshold)
            {
                ncnn::Mat pred_bbox = pred_grid.range(0, reg_max_1 * 4).reshape(reg_max_1, 4).clone();

                // 每个 ltrb 分支做 softmax -> 期望距离
                {
                    ncnn::Layer* softmax = ncnn::create_layer("Softmax");
                    ncnn::ParamDict pd;
                    pd.set(0, 1); // axis
                    pd.set(1, 1);
                    softmax->load_param(pd);

                    ncnn::Option opt;
                    opt.num_threads = 1;
                    opt.use_packing_layout = false;

                    softmax->create_pipeline(opt);
                    softmax->forward_inplace(pred_bbox, opt);
                    softmax->destroy_pipeline(opt);
                    delete softmax;
                }

                float pred_ltrb[4];
                for (int k = 0; k < 4; k++)
                {
                    float dis = 0.f;
                    const float* dis_after_sm = pred_bbox.row(k);
                    for (int l = 0; l < reg_max_1; l++)
                        dis += l * dis_after_sm[l];
                    pred_ltrb[k] = dis * stride;
                }

                float pb_cx = (gx + 0.5f) * stride;
                float pb_cy = (gy + 0.5f) * stride;

                float x0 = pb_cx - pred_ltrb[0];
                float y0 = pb_cy - pred_ltrb[1];
                float x1 = pb_cx + pred_ltrb[2];
                float y1 = pb_cy + pred_ltrb[3];

                Object obj;
                obj.rect.x = x0;
                obj.rect.y = y0;
                obj.rect.width  = x1 - x0;
                obj.rect.height = y1 - y0;
                obj.label = label;
                obj.prob  = score;
                obj.gindex= gy * num_grid_x + gx;

                objects.push_back(obj);
            }
        }
    }
}

void YOLOE::generate_proposals_pyramids(const ncnn::Mat& pred, const std::vector<int>& strides, const ncnn::Mat& in_pad, float prob_threshold, std::vector<Object>& objects)
{
    const int w = in_pad.w;
    const int h = in_pad.h;

    int pred_row_offset = 0;
    for (size_t i = 0; i < strides.size(); i++)
    {
        const int stride = strides[i];

        const int num_grid_x = w / stride;
        const int num_grid_y = h / stride;
        const int num_grid = num_grid_x * num_grid_y;

        std::vector<Object> objects_stride;
        generate_proposals_stride(pred.row_range(pred_row_offset, num_grid), stride, in_pad, prob_threshold, objects_stride);

        for (size_t j = 0; j < objects_stride.size(); j++)
        {
            Object obj = objects_stride[j];
            obj.gindex += pred_row_offset;
            objects.push_back(obj);
        }

        pred_row_offset += num_grid;
    }
}

cv::Mat YOLOE::make_mask(cv::Mat &img, int P3_width, int P3_heigh, cv::Rect object, const int target_size)
{
    // 安全检查
    if (img.empty() || P3_width <= 0 || P3_heigh <= 0 || target_size <= 0) {
        return cv::Mat::zeros(std::max(1, P3_heigh), std::max(1, P3_width), CV_32FC1);
    }

    // 原图尺寸
    const int iw = img.cols;
    const int ih = img.rows;

    // 1) 计算 letterbox 的缩放与填充（与 Ultralytics 一致的思路）
    const float r = std::min(target_size / static_cast<float>(iw), target_size / static_cast<float>(ih));
    const int new_w = static_cast<int>(std::round(iw * r));
    const int new_h = static_cast<int>(std::round(ih * r));
    const float dw = (target_size - new_w) * 0.5f;  // 左右各一半
    const float dh = (target_size - new_h) * 0.5f;  // 上下各一半

    // 2) 将原图坐标下的 bbox 映射到 letterbox 坐标
    float x1 = object.x * r + dw;
    float y1 = object.y * r + dh;
    float x2 = (object.x + object.width)  * r + dw;
    float y2 = (object.y + object.height) * r + dh;

    // 与像素网格对齐（将右下角作为包含边界），并裁剪到 [0, target_size-1]
    auto clampf = [](float v, float lo, float hi) { return std::max(lo, std::min(v, hi)); };
    x1 = clampf(std::floor(x1), 0.0f, target_size - 1.0f);
    y1 = clampf(std::floor(y1), 0.0f, target_size - 1.0f);
    x2 = clampf(std::ceil(x2) - 1.0f, 0.0f, target_size - 1.0f);
    y2 = clampf(std::ceil(y2) - 1.0f, 0.0f, target_size - 1.0f);

    if (x2 < x1 || y2 < y1) {
        // 完全越界或无效，返回全 0
        return cv::Mat::zeros(P3_heigh, P3_width, CV_32FC1);
    }

    // 3) 从 letterbox 尺度映射到 P3 尺度
    const float sx = static_cast<float>(P3_width) / static_cast<float>(target_size);
    const float sy = static_cast<float>(P3_heigh) / static_cast<float>(target_size);

    int px1 = static_cast<int>(std::floor(x1 * sx));
    int py1 = static_cast<int>(std::floor(y1 * sy));
    int px2 = static_cast<int>(std::ceil((x2 + 1.0f) * sx) - 1);  // +1 再缩放再 -1，确保包含右/下边界
    int py2 = static_cast<int>(std::ceil((y2 + 1.0f) * sy) - 1);

    auto clampi = [](int v, int lo, int hi) { return std::max(lo, std::min(v, hi)); };
    px1 = clampi(px1, 0, P3_width - 1);
    py1 = clampi(py1, 0, P3_heigh - 1);
    px2 = clampi(px2, 0, P3_width - 1);
    py2 = clampi(py2, 0, P3_heigh - 1);

    if (px2 < px1 || py2 < py1) {
        return cv::Mat::zeros(P3_heigh, P3_width, CV_32FC1);
    }

    // 4) 在 P3 网格上绘制填充矩形，值为 1.0
    cv::Mat mask = cv::Mat::zeros(P3_heigh, P3_width, CV_32FC1);
    cv::Rect rect_p3(px1, py1, px2 - px1 + 1, py2 - py1 + 1);
    cv::rectangle(mask, rect_p3, cv::Scalar(1.0f), cv::FILLED);

    return mask;
}