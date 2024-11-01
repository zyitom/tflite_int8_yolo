#ifndef _YOLOX_CPP_CORE_HPP
#define _YOLOX_CPP_CORE_HPP

#include <cstdint>
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
        int input_w_ = 416;
        int input_h_ = 416;
        float nms_thresh_;
        float bbox_conf_thresh_;
        int num_classes_;
        bool p6_;
        std::string model_version_;
        // const std::vector<float> mean_ = {0.485, 0.456, 0.406};
        // const std::vector<float> std_ = {0.229, 0.224, 0.225};
        const std::vector<int> strides_ = {8, 16, 32};


        float input_scale_;
        int input_zero_point_;
        float output_scale_;
        int output_zero_point_;

        std::vector<GridAndStride> grid_strides_;




void blobFromImage(const cv::Mat &img, int8_t *blob_data)
{
    const size_t channels = 3;
    const size_t img_h = img.rows;  
    const size_t img_w = img.cols;
    const size_t img_hw = img_h * img_w;

    // 指向各个通道的指针
    int8_t *blob_data_ch0 = blob_data;
    int8_t *blob_data_ch1 = blob_data + img_hw;
    int8_t *blob_data_ch2 = blob_data + img_hw * 2;

    const float scale = 1.0f / 255.0f;
 

    for (size_t i = 0; i < img_hw; ++i)
    {
        const size_t src_idx = i * channels;
        
        // 对每个通道记得减去均值
        float ch0 = img.data[src_idx + 0] -128  ;
        float ch1 = img.data[src_idx + 1] -128 ; 
        float ch2 = img.data[src_idx + 2] -128 ; 

        // 量化到int8
        blob_data_ch0[i] = qnt_f32_to_affine(ch0, input_zero_point_, input_scale_);
        blob_data_ch1[i] = qnt_f32_to_affine(ch1, input_zero_point_, input_scale_);
        blob_data_ch2[i] = qnt_f32_to_affine(ch2, input_zero_point_, input_scale_);


    }
}
void blobFromImage_NHWC(const cv::Mat &img, int8_t *blob_data)
{
    const size_t channels = 3;
    const size_t img_h = img.rows;
    const size_t img_w = img.cols;

    const float scale = 1.0f ;
    const std::vector<float> mean = {0.485f, 0.456f, 0.406f};
    const std::vector<float> std = {0.229f, 0.224f, 0.225f};

    for (size_t h = 0; h < img_h; ++h)
    {
        for (size_t w = 0; w < img_w; ++w)
        {
            const size_t src_idx = (h * img_w + w) * channels;
            const size_t dst_idx = (h * img_w + w) * channels;

            float ch0 = img.data[src_idx + 0] * scale;
            float ch1 = img.data[src_idx + 1] * scale;
            float ch2 = img.data[src_idx + 2] * scale;

            blob_data[dst_idx + 0] = qnt_f32_to_affine(ch0, input_zero_point_, input_scale_);
            blob_data[dst_idx + 1] = qnt_f32_to_affine(ch1, input_zero_point_, input_scale_);
            blob_data[dst_idx + 2] = qnt_f32_to_affine(ch2, input_zero_point_, input_scale_);
        }
    }
}




inline static int32_t __clip(float val, float min, float max)
{
    float f = val <= min ? min : (val >= max ? max : val);
    return f;
}

static int8_t qnt_f32_to_affine(float f32, int32_t zp, float scale)
{
    float dst_val = (f32 / scale) + zp;
    int8_t res = (int8_t)__clip(dst_val, -128, 127);
    return res;
}

static float deqnt_affine_to_f32(int8_t qnt, int32_t zp, float scale) { return ((float)qnt - (float)zp) * scale; }

        cv::Mat static_resize(const cv::Mat &img)
        {
        
            int input_w = input_w_; 
            int input_h = input_h_; 

            float r = std::min(input_w / (img.cols * 1.0), input_h / (img.rows * 1.0));
            int new_w = r * img.cols;
            int new_h = r * img.rows;

        
            cv::Mat resized_img;
            cv::resize(img, resized_img, cv::Size(new_w, new_h));

      
            cv::Mat dst(input_h, input_w, CV_8UC3, cv::Scalar(114, 114, 114));
            resized_img.copyTo(dst(cv::Rect(0, 0, new_w, new_h)));
            cv::imshow("resized", dst);
            cv::waitKey(0);
            return dst;
        }

        // for NCHW
     



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

        void generate_yolox_proposals(const std::vector<GridAndStride> &grid_strides, int8_t *feat_ptr, const float prob_threshold, std::vector<Object> &objects)
        {
            const int num_anchors = grid_strides.size();
            objects.clear();

            for (int anchor_idx = 0; anchor_idx < num_anchors; ++anchor_idx)
            {
                const int grid0 = grid_strides[anchor_idx].grid0;
                const int grid1 = grid_strides[anchor_idx].grid1;
                const int stride = grid_strides[anchor_idx].stride;

                const int basic_pos = anchor_idx * (num_classes_ + 5);

                int class_id = 0;
                float max_class_score = 0.0f;
                
                    float box_objectness = deqnt_affine_to_f32(feat_ptr[basic_pos + 4], output_zero_point_, output_scale_);
                    auto begin = feat_ptr + (basic_pos + 5);
                    auto end = feat_ptr + (basic_pos + 5 + num_classes_);
                    auto max_elem = std::max_element(begin, end);
                    class_id = max_elem - begin;
                    max_class_score = deqnt_affine_to_f32(*max_elem, output_zero_point_, output_scale_) * box_objectness;
                    //float class_score = sigmoid(deqnt_affine_to_f32(*max_elem, output_zero_point_, output_scale_));
         
    
                if (box_objectness > prob_threshold)
                {   
                    // yolox/models/yolo_head.py decode logic
                    //  outputs[..., :2] = (outputs[..., :2] + grids) * strides
                    //  outputs[..., 2:4] = torch.exp(outputs[..., 2:4]) * strides
                    const float x_center = (deqnt_affine_to_f32(feat_ptr[basic_pos + 0], output_zero_point_, output_scale_) + grid0) * stride;
                    const float y_center = (deqnt_affine_to_f32(feat_ptr[basic_pos + 1], output_zero_point_, output_scale_) + grid1) * stride;
                    const float w = exp(deqnt_affine_to_f32(feat_ptr[basic_pos + 2], output_zero_point_, output_scale_)) * stride;
                    const float h = exp(deqnt_affine_to_f32(feat_ptr[basic_pos + 3], output_zero_point_, output_scale_)) * stride;
                    const float x0 = x_center - w * 0.5f;
                    const float y0 = y_center - h * 0.5f;

                    Object obj;
                    obj.rect.x = x0;
                    obj.rect.y = y0;
                    obj.rect.width = w;
                    obj.rect.height = h;
                    obj.label = class_id;
                    obj.prob = max_class_score;
                    objects.push_back(obj);
                }
            } // point anchor loop
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

        void decode_outputs(int8_t *prob, const std::vector<GridAndStride> &grid_strides,
                            std::vector<Object> &objects, const float bbox_conf_thresh,
                            const float scale, const int img_w, const int img_h)
        {

            std::vector<Object> proposals;
            generate_yolox_proposals(grid_strides, prob, bbox_conf_thresh, proposals);

            std::sort(
                proposals.begin(), proposals.end(),
                [](const Object &a, const Object &b)
                {
                    return a.prob > b.prob; // descent
                });

            std::vector<int> picked;
            nms_sorted_bboxes(proposals, picked, nms_thresh_);

            int count = picked.size();
            objects.resize(count);
            const float max_x = static_cast<float>(img_w - 1);
            const float max_y = static_cast<float>(img_h - 1);

            for (int i = 0; i < count; ++i)
            {
                objects[i] = proposals[picked[i]];

                // adjust offset to original unpadded
                float x0 = objects[i].rect.x / scale;
                float y0 = objects[i].rect.y / scale;
                float x1 = (objects[i].rect.x + objects[i].rect.width) / scale;
                float y1 = (objects[i].rect.y + objects[i].rect.height) / scale;

                // clip
                x0 = std::max(std::min(x0, max_x), 0.f);
                y0 = std::max(std::min(y0, max_y), 0.f);
                x1 = std::max(std::min(x1, max_x), 0.f);
                y1 = std::max(std::min(y1, max_y), 0.f);

                objects[i].rect.x = x0;
                objects[i].rect.y = y0;
                objects[i].rect.width = x1 - x0;
                objects[i].rect.height = y1 - y0;
            }
        }
    };

#endif
