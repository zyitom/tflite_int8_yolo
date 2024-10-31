#include "yolox_tflite.hpp"
#include <opencv2/highgui.hpp>


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
            input_scale_ = tensor->params.scale;
            input_zero_point_ = tensor->params.zero_point;
            
            std::cout << "Input quantization params:" << std::endl;
            std::cout << " scale: " << input_scale_ << std::endl;
            std::cout << " zero_point: " << input_zero_point_ << std::endl;
        }
        
        // 获取输出张量的量化参数
        {
            TfLiteTensor* tensor = this->interpreter_->output_tensor(0);
            output_scale_ = tensor->params.scale;
            output_zero_point_ = tensor->params.zero_point;
            
            std::cout << "Output quantization params:" << std::endl;
            std::cout << " scale: " << output_scale_ << std::endl;
            std::cout << " zero_point: " << output_zero_point_ << std::endl;
        }

        // Prepare GridAndStrides


            std::cout << "input_w_: " << this->input_w_ << std::endl;
            std::cout << "input_h_: " << this->input_h_ << std::endl;
            generate_grids_and_stride(this->input_w_, this->input_h_, this->strides_, this->grid_strides_);
        
    }
    YoloXTflite::~YoloXTflite()
    {
        //TfLiteXNNPackDelegateDelete(this->delegate_);
    }
  std::vector<Object> YoloXTflite::inference(const cv::Mat &frame)
{
    std::cout << "\n=== Starting Inference ===" << std::endl;
    
    // 预处理
    cv::Mat pr_img = static_resize(frame);
    std::cout << "Resized image: " << pr_img.size() << std::endl;
    
    // 获取输入张量
    int8_t* input_blob = interpreter_->typed_input_tensor<int8_t>(0);
    if (!input_blob) {
        std::cerr << "Failed to get input tensor!" << std::endl;
        return std::vector<Object>();
    }
    
    // 处理图像
    if(is_nchw_){
        std::cout << "Using NCHW format" << std::endl;
        blobFromImage(pr_img, input_blob);
    } else {
        std::cout << "Using NHWC format" << std::endl;
        blobFromImage_NHWC(pr_img, input_blob);
    }
    
    // 执行推理
    std::cout << "Invoking interpreter..." << std::endl;
    if (interpreter_->Invoke() != kTfLiteOk) {
        std::cerr << "Failed to invoke interpreter" << std::endl;
        return std::vector<Object>();
    }
    
    // 获取输出
    int8_t* output = interpreter_->typed_output_tensor<int8_t>(0);
    if (!output) {
        std::cerr << "Failed to get output tensor!" << std::endl;
        return std::vector<Object>();
    }
    
    // 打印一些输出值
    std::cout << "\n=== Sample Output Values ===" << std::endl;
    for (int i = 0; i < 10; ++i) {
        float dequant_val = deqnt_affine_to_f32(output[i], output_zero_point_, output_scale_);
        std::cout << "Output[" << i << "] = " << (int)output[i] 
                 << " (dequantized: " << dequant_val << ")" << std::endl;
    }
    // 检查模型是否正确加载
std::cout << "Model input shape: " 
          << interpreter_->input_tensor(0)->dims->data[0] << "x"
          << interpreter_->input_tensor(0)->dims->data[1] << "x"
          << interpreter_->input_tensor(0)->dims->data[2] << "x"
          << interpreter_->input_tensor(0)->dims->data[3] << std::endl;

// 检查量化参数
std::cout << "Input scale: " << input_scale_ << std::endl;
std::cout << "Input zero point: " << input_zero_point_ << std::endl;
std::cout << "Output scale: " << output_scale_ << std::endl;
std::cout << "Output zero point: " << output_zero_point_ << std::endl;
    // 后处理
    std::vector<Object> objects;
    const float scale = std::min(input_w_ / (float)frame.cols, input_h_ / (float)frame.rows);
    decode_outputs(output, grid_strides_, objects, bbox_conf_thresh_, scale, frame.cols, frame.rows);
    
    std::cout << "Found " << objects.size() << " objects" << std::endl;
    
    return objects;
}


int main(int argc, char* argv[]) {
    try {
        // 配置参数
        const std::string model_path = "/home/zyi/Downloads/yolox_nano_ti_lite/yolox_nano_ti_lite_full_integer_quant.tflite";  // 需要替换为实际的模型路径
        //const char* model_path = "/home/zyi/Downloads/yolox_nano.tflite";  // 需要替换为实际的模型路径
        const int num_threads = 4;                    // 线程数
        const float nms_threshold = 0.45f;            // NMS阈值
        const float confidence_threshold = 0.3f;      // 置信度阈值
        const std::string model_version = "0.1.1rc0";    // 模型版本
        const int num_classes = 80;                   // COCO数据集类别数
        const bool p6 = false;                        // 是否使用P6
        const bool is_nchw = false;                   // 是否使用NCHW格式
        
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
