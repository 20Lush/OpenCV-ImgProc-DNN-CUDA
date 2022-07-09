#include "Analysis.hpp"

// 7/6/2022 ---------------------
// -> Serial interfacing
// -> Get the arduino and exe talking to eachother
// -> automatic port searching? handshake routine probable

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

    string aim_lab = "aimlab_model/";

    ifstream ifs(string(BASE_PATH + aim_lab + "coco.names").c_str());

    // +----------------------------------------------------+
    #pragma endregion file_pathing

    vector<string> CLASS_NAMES;
    string line;
    while(getline(ifs, line))
        CLASS_NAMES.push_back(line);

    #pragma region dnn_spinup // dnn resource targeting and net object instantiation
    // +----------------------------------------------------+

    auto net = cv::dnn::readNetFromDarknet(BASE_PATH + aim_lab + "custom-yolov4-tiny-detector.cfg", BASE_PATH + aim_lab + "custom-yolov4-tiny-detector_best.weights");

    net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);

    net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);

    // +----------------------------------------------------+
    #pragma endregion dnn_spinup

    cv::Mat frame; // moving this anywhere else produces catastrophic memory leak when detections are made. leak is observed to bleed into the values of the center Point found in a high prio det box
                   // i.e leave this the hell alone
    for(;;){

        cap >> frame;
        analysis.DET_COUNT = 0;

        #pragma region image_mutation // CV cropping through GPU routine. Output is var "croppedImage"
        // +----------------------------------------------------+

        cv::cuda::GpuMat gpu_fullImage;

        gpu_fullImage.upload(frame);

        cv::cuda::GpuMat gpu_croppedImage = gpu_fullImage(analysis.getCenterSquare(analysis.IMG_WIDTH, analysis.IMG_HEIGHT, analysis.INPUT_WIDTH)); // more for offsetting system resources than any percieved performance gained. Its cool and I wanted to do it ok
        
        cv::Mat croppedImage; 

        gpu_croppedImage.download(croppedImage);

        // +----------------------------------------------------+
        #pragma endregion image_mutation 

        #pragma region neural_net_impl // blob from croppedImage, outputs are stored in a Mat vector "outputs" 
        // +----------------------------------------------------+

        cv::Mat blob = blobFromImage(croppedImage, 1/255.0, cv::Size(analysis.INPUT_WIDTH, analysis.INPUT_HEIGHT), cv::Scalar(0,0,0), true, false);

        net.setInput(blob);

        vector<cv::Mat> outputs;

        net.forward(outputs, analysis.getOutputNames(net));

        // +----------------------------------------------------+
        #pragma endregion neural_net_impl

        Point target; // (x,y) off the postProcess' highest priority detection is put here. (0,0) when there are no detections
        analysis.postProcess(croppedImage, outputs, CLASS_NAMES, &target); // has box drawing embedded into it
        analysis.drawDetectionCount(croppedImage); // top left counter

        imshow("frame", croppedImage);

        if(waitKey(1) == 'q')
            break;
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;

}

