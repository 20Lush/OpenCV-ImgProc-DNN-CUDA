#include "Analysis.hpp"


int main(int, char**) {

    Analysis analysis;
    

    VideoCapture cap;
    cap.open(0);
    cap.set(CAP_PROP_FRAME_WIDTH, analysis.IMG_WIDTH); //could definitally find all of these esoteric values through hwnd
    cap.set(CAP_PROP_FRAME_HEIGHT, analysis.IMG_HEIGHT);
    cap.set(CAP_PROP_FPS, 60);
    cap.set(CAP_PROP_AUTO_EXPOSURE, false);

    string BASE_PATH = "C:/Users/zachf/Documents/.Computer Vision/opencv-gpu-test/"; //need to eventually find a more portable way to do this (user prompt maybe?)
    string valorant = "non_descript_game_model/"; //i hate file systems

    vector<string> CLASS_NAMES;
    ifstream ifs(string(BASE_PATH + valorant + "coco-dataset.labels").c_str());
    string line;
    while(getline(ifs, line))
        CLASS_NAMES.push_back(line);

    auto net = readNetFromDarknet(BASE_PATH + valorant + "yolov4-tiny.cfg", BASE_PATH + valorant + "yolov4-tiny.weights");
    net.setPreferableBackend(DNN_BACKEND_CUDA);
    net.setPreferableTarget(DNN_TARGET_CUDA);

    Mat frame;
    for(;;){

        cap >> frame;
        analysis.DET_COUNT = 0;

        cuda::GpuMat gpu_fullImage;
        gpu_fullImage.upload(frame);

        cuda::GpuMat gpu_croppedImage = gpu_fullImage(analysis.getCenterSquare(analysis.IMG_WIDTH, analysis.IMG_HEIGHT, analysis.INPUT_WIDTH)); // more for offsetting system resources than any percieved performance gained. Its cool and I wanted to do it ok
        
        Mat croppedImage; 
        gpu_croppedImage.download(croppedImage);
        
        Mat blob = blobFromImage(croppedImage, 1/255.0, Size(analysis.INPUT_WIDTH, analysis.INPUT_HEIGHT), Scalar(0,0,0), true, false);
        net.setInput(blob);

        vector<Mat> outputs;
        net.forward(outputs, analysis.getOutputNames(net));

        analysis.postProcess(croppedImage, outputs, CLASS_NAMES);
        analysis.drawDetectionCount(croppedImage);
        imshow("frame", croppedImage);
        

        if(waitKey(1) == 'q')
            break;

    }

    cap.release();
    destroyAllWindows();
    return 0;

}

