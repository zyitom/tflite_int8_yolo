#ifndef _YOLOX_CPP_CORE_HPP
#define _YOLOX_CPP_CORE_HPP

#include <opencv2/core/types.hpp>
#include <opencv2/opencv.hpp>

/**
 * @brief Define names based depends on Unicode path support
 */
#define file_name_t std::string

    struct Object
    {
        cv::Rect_<float> rect;
        int label;
        float prob;
    };

    struct GridAndStride
    {
        int grid0;
        int grid1;
        int stride;
        GridAndStride(const int grid0_, const int grid1_, const int stride_)
            : grid0(grid0_), grid1(grid1_), stride(stride_)
        {
        }
    };

    class AbcYoloX
    {
    public:
        AbcYoloX() {}
        AbcYoloX(float nms_th = 0.45, float conf_th = 0.3,
                 const std::string &model_version = "0.1.1rc0",
                 int num_classes = 80, bool p6 = false)
            : nms_thresh_(nms_th), bbox_conf_thresh_(conf_th),
              num_classes_(num_classes), p6_(p6), model_version_(model_version)
        {
        }
        virtual std::vector<Object> inference(const cv::Mat &frame) = 0;

    protected:
        int input_w_;
        int input_h_;
        float nms_thresh_;
        float bbox_conf_thresh_;
        int num_classes_;
        bool p6_;
        std::string model_version_;
        float input_scale_;
        int input_zero_point_;
        float output_scale_;
        int output_zero_point_;

        const std::vector<int> strides_ = {8, 16, 32};
        const std::vector<int> strides_p6_ = {8, 16, 32, 64};
        std::vector<GridAndStride> grid_strides_;

        cv::Mat static_resize(const cv::Mat &img)
        {
            const float r = std::min(
                static_cast<float>(input_w_) / static_cast<float>(img.cols),
                static_cast<float>(input_h_) / static_cast<float>(img.rows));
            const int unpad_w = r * img.cols;
            const int unpad_h = r * img.rows;
            cv::Mat re(unpad_h, unpad_w, CV_8UC3);
            cv::resize(img, re, re.size());
            cv::Mat out(input_h_, input_w_, CV_8UC3, cv::Scalar(114, 114, 114));
            re.copyTo(out(cv::Rect(0, 0, re.cols, re.rows)));
            return out;
        }

        // for NCHW
void blobFromImage(const cv::Mat &img, int8_t *blob_data)
{
    const size_t channels = 3;
    const size_t img_h = img.rows;
    const size_t img_w = img.cols;
    const size_t img_hw = img_h * img_w;
    
    int8_t *blob_data_ch0 = blob_data;
    int8_t *blob_data_ch1 = blob_data + img_hw;
    int8_t *blob_data_ch2 = blob_data + img_hw * 2;
    
    std::cout << "Quantization params - scale: " << input_scale_ 
              << ", zero_point: " << input_zero_point_ << std::endl;
              
    for (size_t i = 0; i < img_hw; ++i)
    {
        const size_t src_idx = i * channels;
        
        // 获取原始像素值
        float r = static_cast<float>(img.data[src_idx + 0]);
        float g = static_cast<float>(img.data[src_idx + 1]);
        float b = static_cast<float>(img.data[src_idx + 2]);
        
        // 正确的量化公式：q = round(x / scale) + zero_point
        // 其中x是归一化后的值 [0,1]
        float r_norm = r / 255.0f;
        float g_norm = g / 255.0f;
        float b_norm = b / 255.0f;
        
        int32_t qr = static_cast<int32_t>(std::round(r_norm / input_scale_)) + input_zero_point_;
        int32_t qg = static_cast<int32_t>(std::round(g_norm / input_scale_)) + input_zero_point_;
        int32_t qb = static_cast<int32_t>(std::round(b_norm / input_scale_)) + input_zero_point_;
        
        // 限制在int8范围内
        blob_data_ch0[i] = std::clamp(qr, -128, 127);
        blob_data_ch1[i] = std::clamp(qg, -128, 127);
        blob_data_ch2[i] = std::clamp(qb, -128, 127);
        
        // 打印第一个像素的详细计算过程
        if (i == 340) {
            std::cout << "\nFirst pixel quantization details:" << std::endl;
            std::cout << "Original values - R: " << r << ", G: " << g << ", B: " << b << std::endl;
            std::cout << "Normalized values - R: " << r_norm << ", G: " << g_norm << ", B: " << b_norm << std::endl;
            std::cout << "Intermediate values (before zero_point) - R: " << std::round(r_norm / input_scale_) 
                      << ", G: " << std::round(g_norm / input_scale_) 
                      << ", B: " << std::round(b_norm / input_scale_) << std::endl;
            std::cout << "Final quantized values - R: " << (int)blob_data_ch0[i] 
                      << ", G: " << (int)blob_data_ch1[i] 
                      << ", B: " << (int)blob_data_ch2[i] << std::endl;
            
            // 验证反量化
            float dequant_r = (static_cast<float>(blob_data_ch0[i]) - input_zero_point_) * input_scale_;
            std::cout << "Dequantized R: " << dequant_r << " (should be close to " << r_norm << ")" << std::endl;
        }
    }
}

// 2. 添加量化辅助函数
inline int8_t quantize_to_int8(float value, float scale, int zero_point)
{
    int32_t quantized = std::round(value / scale) + zero_point;
    return static_cast<int8_t>(std::clamp(quantized, -128, 127));
}


        void generate_grids_and_stride(const int target_w, const int target_h, const std::vector<int> &strides, std::vector<GridAndStride> &grid_strides)
        {
            grid_strides.clear();
            for (auto stride : strides)
            {
                const int num_grid_w = target_w / stride;
                const int num_grid_h = target_h / stride;
                for (int g1 = 0; g1 < num_grid_h; ++g1)
                {
                    for (int g0 = 0; g0 < num_grid_w; ++g0)
                    {
                        grid_strides.emplace_back(g0, g1, stride);
                    }
                }
            }
        }

inline int8_t qnt_f32_to_affine(float threshold, int32_t zp, float scale) {
    return static_cast<int8_t>(std::round(threshold / scale) + zp);
}

// 添加反量化函数
inline float deqnt_affine_to_f32(int8_t value, int32_t zp, float scale) {
    return (static_cast<float>(value) - zp) * scale;
}

// 修改generate_yolox_proposals函数
void generate_yolox_proposals(
    const std::vector<GridAndStride> &grid_strides,
    const int8_t *feat_ptr,
    const float prob_threshold,
    std::vector<Object> &objects)
{
    objects.clear();
    
    // 将浮点阈值转换为int8阈值
    int8_t threshold_i8 = qnt_f32_to_affine(prob_threshold, output_zero_point_, output_scale_);
    
    // 计算每个特征图的大小
    for(auto stride : strides_) {
        int grid_h = input_h_ / stride;
        int grid_w = input_w_ / stride;
        int grid_len = grid_h * grid_w;
        
        std::cout << "Processing grid: " << grid_h << "x" << grid_w << " stride: " << stride << std::endl;
        
        // 遍历网格
        for(int i = 0; i < grid_h; i++) {
            for(int j = 0; j < grid_w; j++) {
                int offset = i * grid_w + j;
                
                // 先检查objectness分数
                int8_t box_confidence = feat_ptr[4 * grid_len + offset];
                if(box_confidence >= threshold_i8) {
                    // 查找最大类别分数
                    const int8_t* scores_ptr = feat_ptr + 5 * grid_len + offset;
                    int8_t max_score = scores_ptr[0];
                    int max_class_id = 0;
                    
                    for(int c = 1; c < num_classes_; c++) {
                        int8_t score = scores_ptr[c * grid_len];
                        if(score > max_score) {
                            max_score = score;
                            max_class_id = c;
                        }
                    }
                    
                    // 如果类别分数也超过阈值
                    if(max_score >= threshold_i8) {
                        // 解码box坐标
                        const int8_t* box_ptr = feat_ptr + offset;
                        float x = deqnt_affine_to_f32(box_ptr[0], output_zero_point_, output_scale_);
                        float y = deqnt_affine_to_f32(box_ptr[grid_len], output_zero_point_, output_scale_);
                        float w = deqnt_affine_to_f32(box_ptr[2 * grid_len], output_zero_point_, output_scale_);
                        float h = deqnt_affine_to_f32(box_ptr[3 * grid_len], output_zero_point_, output_scale_);
                        
                        // 计算中心点坐标
                        x = (x + j) * stride;
                        y = (y + i) * stride;
                        // 计算宽高
                        w = std::exp(w) * stride;
                        h = std::exp(h) * stride;
                        
                        // 转换为左上角坐标
                        x -= (w / 2.0f);
                        y -= (h / 2.0f);
                        
                        // 计算最终的置信度分数
                        float confidence = deqnt_affine_to_f32(box_confidence, output_zero_point_, output_scale_);
                        float class_score = deqnt_affine_to_f32(max_score, output_zero_point_, output_scale_);
                        float final_score = confidence * class_score;
                        
                        if(i == 0 && j == 0) {  // 打印第一个检测框的调试信息
                            std::cout << "Debug box:" << std::endl;
                            std::cout << "  raw confidence: " << (int)box_confidence 
                                      << " -> " << confidence << std::endl;
                            std::cout << "  raw class score: " << (int)max_score 
                                      << " -> " << class_score << std::endl;
                            std::cout << "  final score: " << final_score << std::endl;
                            std::cout << "  box: " << x << "," << y << "," << w << "," << h << std::endl;
                        }
                        
                        // 添加到结果中
                        Object obj;
                        obj.rect.x = x;
                        obj.rect.y = y;
                        obj.rect.width = w;
                        obj.rect.height = h;
                        obj.label = max_class_id;
                        obj.prob = final_score;
                        objects.push_back(obj);
                    }
                }
            }
        }
    }
    
    std::cout << "Found " << objects.size() << " objects" << std::endl;
}
// 添加sigmoid辅助函数
inline float sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

// 4. 添加反量化辅助函数
inline float dequantize_int8(int8_t value, float scale, int zero_point)
{
    return static_cast<float>(value - zero_point) * scale;
}

        float intersection_area(const Object &a, const Object &b)
        {
            const cv::Rect_<float> inter = a.rect & b.rect;
            return inter.area();
        }

        void nms_sorted_bboxes(const std::vector<Object> &faceobjects, std::vector<int> &picked, const float nms_threshold)
        {
            picked.clear();

            const int n = faceobjects.size();

            std::vector<float> areas(n);
            for (int i = 0; i < n; ++i)
            {
                areas[i] = faceobjects[i].rect.area();
            }

            for (int i = 0; i < n; ++i)
            {
                const Object &a = faceobjects[i];
                const int picked_size = picked.size();

                bool keep = true;
                for (int j = 0; j < picked_size; ++j)
                {
                    const Object &b = faceobjects[picked[j]];

                    // intersection over union
                    const float inter_area = intersection_area(a, b);
                    const float union_area = areas[i] + areas[picked[j]] - inter_area;
                    // float IoU = inter_area / union_area
                    if (inter_area / union_area > nms_threshold)
                    {
                        keep = false;
                        break;
                    }
                }

                if (keep)
                    picked.push_back(i);
            }
        }

void decode_outputs(
    const int8_t *prob,
    const std::vector<GridAndStride> &grid_strides,
    std::vector<Object> &objects,
    const float bbox_conf_thresh,
    const float scale,
    const int img_w,
    const int img_h)
{
    std::vector<Object> proposals;
    generate_yolox_proposals(grid_strides, prob, bbox_conf_thresh, proposals);
    
    std::cout << "Proposals before NMS: " << proposals.size() << std::endl;
    
    std::sort(proposals.begin(), proposals.end(),
              [](const Object& a, const Object& b) { return a.prob > b.prob; });

    std::vector<int> picked;
    nms_sorted_bboxes(proposals, picked, nms_thresh_);

    int count = picked.size();
    objects.resize(count);
    
    // 调整到原始图像尺寸
    for (int i = 0; i < count; i++) {
        objects[i] = proposals[picked[i]];
        
        // 缩放到原始图像大小
        float scale_x = static_cast<float>(img_w) / input_w_;
        float scale_y = static_cast<float>(img_h) / input_h_;
        
        objects[i].rect.x *= scale_x;
        objects[i].rect.y *= scale_y;
        objects[i].rect.width *= scale_x;
        objects[i].rect.height *= scale_y;
        
        // 确保框在图像范围内
        objects[i].rect.x = std::max(0.0f, objects[i].rect.x);
        objects[i].rect.y = std::max(0.0f, objects[i].rect.y);
        objects[i].rect.width = std::min(objects[i].rect.width, 
                                       static_cast<float>(img_w) - objects[i].rect.x);
        objects[i].rect.height = std::min(objects[i].rect.height, 
                                        static_cast<float>(img_h) - objects[i].rect.y);
    }
    
    std::cout << "Objects after NMS: " << objects.size() << std::endl;
}
    };

#endif
