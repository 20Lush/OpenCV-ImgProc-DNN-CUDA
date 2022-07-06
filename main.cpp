#include "Analysis.hpp"


int main(int, char**) {

    Analysis analysis;

    #pragma region video_capture_with_properties // cv::VideoCapture setup. [] OBJ: "cap" []
    // +----------------------------------------------------+

    cv::VideoCapture cap;

    cap.open(0);

    cap.set(cv::CAP_PROP_FRAME_WIDTH, analysis.IMG_WIDTH); //could definitally find all of these esoteric values through hwnd

    cap.set(cv::CAP_PROP_FRAME_HEIGHT, analysis.IMG_HEIGHT);

    cap.set(cv::CAP_PROP_FPS, 60);

    cap.set(cv::CAP_PROP_AUTO_EXPOSURE, false);

    // +----------------------------------------------------+
    #pragma endregion video_capture_with_properties

    #pragma region file_pathing // file paths stored as strings. 
    // +----------------------------------------------------+

    string BASE_PATH = "C:/Users/zachf/Documents/.Computer Vision/opencv-gpu-test/"; //need to eventually find a more portable way to do this (user prompt maybe?)

    string valorant = "non_descript_game_model/"; //i hate file systems

    ifstream ifs(string(BASE_PATH + valorant + "coco-dataset.labels").c_str());

    // +----------------------------------------------------+
    #pragma endregion file_pathing

    vector<string> CLASS_NAMES;
    string line;
    while(getline(ifs, line))
        CLASS_NAMES.push_back(line);

    #pragma region dnn_spinup
    // +----------------------------------------------------+

    auto net = cv::dnn::readNetFromDarknet(BASE_PATH + valorant + "yolov4-tiny.cfg", BASE_PATH + valorant + "yolov4-tiny.weights");

    net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);

    net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);

    // +----------------------------------------------------+
    #pragma endregion dnn_spinup

    cv::Mat frame;
    for(;;){

        cap >> frame;
        analysis.DET_COUNT = 0;

        #pragma region image_mutation 
        // +----------------------------------------------------+

        cv::cuda::GpuMat gpu_fullImage;

        gpu_fullImage.upload(frame);

        cv::cuda::GpuMat gpu_croppedImage = gpu_fullImage(analysis.getCenterSquare(analysis.IMG_WIDTH, analysis.IMG_HEIGHT, analysis.INPUT_WIDTH)); // more for offsetting system resources than any percieved performance gained. Its cool and I wanted to do it ok
        
        cv::Mat croppedImage; 

        gpu_croppedImage.download(croppedImage);
        
        cv::Mat blob = blobFromImage(croppedImage, 1/255.0, cv::Size(analysis.INPUT_WIDTH, analysis.INPUT_HEIGHT), cv::Scalar(0,0,0), true, false);

        net.setInput(blob);

        // +----------------------------------------------------+
        #pragma endregion image_mutation 

        #pragma region neural_net_impl
        // +----------------------------------------------------+

        vector<cv::Mat> outputs;

        net.forward(outputs, analysis.getOutputNames(net));

        // +----------------------------------------------------+
        #pragma endregion neural_net_impl

        //instantiate a Rect for which postProcess will transfer the box[closest_idx] to
        Point target; 
        analysis.postProcess(croppedImage, outputs, CLASS_NAMES, &target); // -> push box[closest_dx] into rect
        analysis.drawDetectionCount(croppedImage);

        imshow("frame", croppedImage);

    }

    cap.release();
    cv::destroyAllWindows();
    return 0;

}

