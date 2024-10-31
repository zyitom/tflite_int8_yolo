#include "yolox_tflite.hpp"
#include <opencv2/highgui.hpp>

 void printTensorInfo(TfLiteTensor* tensor, const char* tensor_type) {
    std::cout << tensor_type << " tensor info:" << std::endl;
    std::cout << " name: " << tensor->name << std::endl;
    std::cout << " type: " << tensor->type << std::endl;
    std::cout << " scale: " << tensor->params.scale << std::endl;
    std::cout << " zero_point: " << tensor->params.zero_point << std::endl;
    std::cout << " dims: [";
    for (int i = 0; i < tensor->dims->size; i++) {
        std::cout << tensor->dims->data[i];
        if (i < tensor->dims->size - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
}

    YoloXTflite::YoloXTflite(const file_name_t &path_to_model, int num_threads,
                             float nms_th, float conf_th, const std::string &model_version,
                             int num_classes, bool p6, bool is_nchw)
        : AbcYoloX(nms_th, conf_th, model_version, num_classes, p6), is_nchw_(is_nchw)
    {
        TfLiteStatus status;
        this->model_ = tflite::FlatBufferModel::BuildFromFile(path_to_model.c_str());
        TFLITE_MINIMAL_CHECK(model_);

        this->resolver_ = std::make_unique<tflite::ops::builtin::BuiltinOpResolver>();
        this->interpreter_ = std::make_unique<tflite::Interpreter>();

        tflite::InterpreterBuilder builder(*model_, *this->resolver_);
        builder(&this->interpreter_);
        TFLITE_MINIMAL_CHECK(this->interpreter_ != nullptr);

        TFLITE_MINIMAL_CHECK(this->interpreter_->AllocateTensors() == kTfLiteOk);
        // tflite::PrintInterpreterState(this->interpreter_.get());

        status = this->interpreter_->SetNumThreads(num_threads);
        if (status != TfLiteStatus::kTfLiteOk)
        {
            std::string msg = "Failed to SetNumThreads.";
            throw std::runtime_error(msg.c_str());
        }


        if (this->interpreter_->AllocateTensors() != TfLiteStatus::kTfLiteOk)
        {
            std::string msg = "Failed to allocate tensors.";
            throw std::runtime_error(msg.c_str());
        }
        {
            TfLiteTensor* tensor = this->interpreter_->input_tensor(0);
            printTensorInfo(tensor, "Input");
            
            if (tensor->type != kTfLiteInt8) {
                throw std::runtime_error("Input tensor is not INT8");
            }
            
            input_scale_ = tensor->params.scale;
            input_zero_point_ = tensor->params.zero_point;
            
            // NCHW
            this->input_h_ = tensor->dims->data[2];
            this->input_w_ = tensor->dims->data[3];
        }

        {
            TfLiteTensor* tensor = this->interpreter_->output_tensor(0);
            printTensorInfo(tensor, "Output");
            
            if (tensor->type != kTfLiteInt8) {
                throw std::runtime_error("Output tensor is not INT8");
            }
            
            output_scale_ = tensor->params.scale;
            output_zero_point_ = tensor->params.zero_point;
            this->output_size_ = 1;
            for (int i = 0; i < tensor->dims->size; i++) {
                this->output_size_ *= tensor->dims->data[i];
            }
        }
            generate_grids_and_stride(this->input_w_, this->input_h_, this->strides_, this->grid_strides_);
    }
    YoloXTflite::~YoloXTflite()
    {
        //TfLiteXNNPackDelegateDelete(this->delegate_);
    }

std::vector<Object> YoloXTflite::inference(const cv::Mat &frame) {
    if (frame.empty()) {
        std::cerr << "Input frame is empty!" << std::endl;
        return std::vector<Object>();
    }
    
    // 打印调试信息
    std::cout << "Processing frame: " << frame.size() << ", channels: " << frame.channels() << std::endl;
    
    // preprocess
    cv::Mat pr_img = static_resize(frame);
    std::cout << "Resized image: " << pr_img.size() << std::endl;
    
    // 获取INT8输入张量
    int8_t *input_blob = this->interpreter_->typed_input_tensor<int8_t>(0);
    if (input_blob == nullptr) {
        std::cerr << "Failed to get input tensor!" << std::endl;
        return std::vector<Object>();
    }
    
    // 预处理并量化
    blobFromImage(pr_img, input_blob);
    
    // 打印第一个像素的量化值进行验证
    std::cout << "First pixel quantized value: " << (int)input_blob[0] << std::endl;
    
    // inference
    TfLiteStatus ret = this->interpreter_->Invoke();
    if (ret != TfLiteStatus::kTfLiteOk) {
        std::cerr << "Failed to invoke!" << std::endl;
        return std::vector<Object>();
    }
    
    // 获取INT8输出张量
 const int8_t* output_tensor = this->interpreter_->typed_output_tensor<int8_t>(0);
    if (output_tensor == nullptr) {
        std::cerr << "Failed to get output tensor!" << std::endl;
        return std::vector<Object>();
    }
    
    // 打印前几个输出值
    std::cout << "\nOutput tensor first few values:" << std::endl;
    for(int i = 0; i < 20; i++) {
        float dequant_val = (output_tensor[i] - output_zero_point_) * output_scale_;
        std::cout << "Output[" << i << "]: raw=" << (int)output_tensor[i] 
                  << " dequant=" << dequant_val << std::endl;
    }
    
    // 计算缩放因子
    const float scale = std::min(
        static_cast<float>(this->input_w_) / static_cast<float>(frame.cols),
        static_cast<float>(this->input_h_) / static_cast<float>(frame.rows)
    );
    
    // 直接使用decode_outputs处理INT8输出
    std::vector<Object> objects;
    decode_outputs(
        output_tensor,
        this->grid_strides_,
        objects,
        this->bbox_conf_thresh_,
        scale,
        frame.cols,
        frame.rows
    );
    
    std::cout << "Detected " << objects.size() << " objects" << std::endl;
    return objects;
}


int main(int argc, char* argv[]) {
    try {
        // 配置参数
        const std::string model_path = "/home/zyi/Downloads/yolox_nano_int8.tflite";  // 需要替换为实际的模型路径
        //const char* model_path = "/home/zyi/Downloads/yolox_nano.tflite";  // 需要替换为实际的模型路径
        const int num_threads = 4;                    // 线程数
        const float nms_threshold = 0.45f;            // NMS阈值
        const float confidence_threshold = 0.3f;      // 置信度阈值
        const std::string model_version = "0.1.1rc0";    // 模型版本
        const int num_classes = 80;                   // COCO数据集类别数
        const bool p6 = false;                        // 是否使用P6
        const bool is_nchw = true;                   // 是否使用NCHW格式
        
        // 初始化YoloX检测器
        YoloXTflite detector(
            model_path,
            num_threads,
            nms_threshold,
            confidence_threshold,
            model_version,
            num_classes,
            p6,
            is_nchw
        );
        
        // 读取图像
        cv::Mat frame = cv::imread("/home/zyi/miniconda3/envs/new/lib/python3.10/site-packages/ultralytics/assets/zidane.jpg");  // 需要替换为实际的图像路径
        if (frame.empty()) {
            std::cerr << "Error: Unable to read the image!" << std::endl;
            return -1;
        }
        
        // 执行检测
        std::vector<Object> detected_objects = detector.inference(frame);
        
        // 在图像上绘制检测结果
        for (const auto& obj : detected_objects) {
            // 绘制边界框
            cv::rectangle(frame, obj.rect, cv::Scalar(0, 255, 0), 2);
            
            // 绘制标签和置信度
            std::string label = std::to_string(obj.label) + ": " + 
                              std::to_string(static_cast<int>(obj.prob * 100)) + "%";
            int baseline = 0;
            cv::Size label_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 
                                                0.5, 1, &baseline);
            cv::putText(frame, label, 
                       cv::Point(obj.rect.x, obj.rect.y - 5),
                       cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
        }
        
        // 显示结果
        cv::imshow("YoloX Detection", frame);
        cv::waitKey(0);
        
        // 保存结果（可选）
        cv::imwrite("detection_result.jpg", frame);
        
        std::cout << "Detection completed. Found " << detected_objects.size() 
                  << " objects." << std::endl;
                  
        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }
}
