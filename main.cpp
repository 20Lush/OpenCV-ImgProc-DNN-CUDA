#include "misc-functions.hpp"

constexpr int IMG_WIDTH = 2560;
constexpr int IMG_HEIGHT = 1440;


int main(int, char**) {

    string BASE_PATH = "C:/";
    string valorant = ".val-weights/";

    VideoCapture cap;
    cap.open(0);
    cap.set(CAP_PROP_FRAME_WIDTH, IMG_WIDTH);
    cap.set(CAP_PROP_FRAME_HEIGHT, IMG_HEIGHT);
    cap.set(CAP_PROP_FPS, 60);
    cap.set(CAP_PROP_AUTO_EXPOSURE, false);

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

        cuda::GpuMat gpu_fullImage;
        gpu_fullImage.upload(frame);

        cuda::GpuMat gpu_croppedImage = gpu_fullImage(getCenterSquare(IMG_WIDTH, IMG_HEIGHT, INPUT_WIDTH)); // more for offsetting system resources than any percieved performance gained. Its cool and I wanted to do it ok
        
        Mat croppedImage; 
        gpu_croppedImage.download(croppedImage);
        
        Mat blob = blobFromImage(croppedImage, 1/255.0, Size(INPUT_WIDTH, INPUT_HEIGHT), Scalar(0,0,0), true, false);
        net.setInput(blob);

        vector<Mat> outputs;
        net.forward(outputs, getOutputNames(net));

        postProcess(croppedImage, outputs, CLASS_NAMES);

        imshow("frame", croppedImage);
        

        if(waitKey(1) == 'q')
            break;

    }

    cap.release();
    destroyAllWindows();
    return 0;

}

