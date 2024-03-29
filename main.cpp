#include <config.h>
#include <Analysis.hpp>
#include "ArduSerial.cpp"

int dx=0, dy=0, x_last=0, y_last=0;

int main(int, char**) {

    Analysis analysis;

    #pragma region video_cap_and_NN // a lot of stuff in here
    // +===============================================================================+
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

    string BASE_PATH = "C:/Users/zachf/Documents/.Computer Vision/opencv-gpu-test/models/"; //need to eventually find a more portable way to do this (user prompt maybe?)

    string valorant = "non_descript_game_model/"; //i hate file systems

    string aim_lab = "aimlab_model/";

    ifstream ifs(string(BASE_PATH + aim_lab + "coco.names").c_str());

    // +----------------------------------------------------+
    #pragma endregion file_pathing

    #pragma region class_name_pull
    vector<string> CLASS_NAMES;
    string line;
    while(getline(ifs, line))
        CLASS_NAMES.push_back(line);
    #pragma endregion class_name_pull

    #pragma region dnn_spinup // dnn resource targeting and net object instantiation
    
    // +----------------------------------------------------+

    auto net = cv::dnn::readNetFromDarknet(BASE_PATH + aim_lab + "custom-yolov4-tiny-detector.cfg", BASE_PATH + aim_lab + "custom-yolov4-tiny-detector_final.weights");

    net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);

    net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);

    // +----------------------------------------------------+
    #pragma endregion dnn_spinup
    // +===============================================================================+
    #pragma endregion video_cap_and_NN
    
    #pragma region arduino_serial_setup

    int port_num = 8; // can be automated

    WindowsSerial* port_ptr = s_Ports[port_num];
    serialSetup(port_ptr);

    #pragma endregion arduino_serial_setup

    Point target; // (x,y) off the postProcess' highest priority detection is put here. (-1,-1) when there are no detections
    float dist;
    string packet;
    cv::Mat frame; // moving this anywhere else produces catastrophic memory leak when detections are made. leak is observed to bleed into the values of the center Point found in a high prio det box
                   // i.e leave this the hell alone

    for(;;){

        cap >> frame;
        analysis.DET_COUNT = 0;

        #pragma region image_mutation // CV croppin22g through GPU routine. Output is var "croppedImage"
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

        analysis.postProcess(croppedImage, outputs, CLASS_NAMES, &target, &dist); // has box drawing embedded into it
        analysis.drawDetectionCount(croppedImage); // top left counter

        #if OUTPUT_HEADLESS_MODE

        if(dist < 100){
            packet = to_string(target.x) + ':' + to_string(target.y);
            serialSend(port_ptr, packet);
        }
        
        #else

        packet = to_string(target.x) + ':' + to_string(target.y);
        serialEchoFast(port_ptr, packet);

        imshow("frame", croppedImage);

        if(waitKey(1) == 'q'){
            break;
        }

        #endif



    }

    cap.release();
    cv::destroyAllWindows();
    return 0;

}

