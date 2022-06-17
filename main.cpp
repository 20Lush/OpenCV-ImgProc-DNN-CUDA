#include "misc-functions.hpp"


int main(int, char**) {

    string BASE_PATH = "C:/Users/zachf/Documents/.Computer Vision/opencv-gpu-test/";
    string valorant = "valorant_model/";

    VideoCapture cap = VideoCapture(0);


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

        Mat blob = blobFromImage(frame, 1/255.0, Size(INPUT_WIDTH, INPUT_HEIGHT), Scalar(0,0,0), true, false);

        net.setInput(blob);

        vector<Mat> outputs;
        net.forward(outputs, getOutputNames(net));

        postprocess(frame, outputs, CLASS_NAMES);

        imshow("frame", frame);

        if(waitKey(1) == 'q')
            break;

    }

    cap.release();
    destroyAllWindows();
    return 0;

}
