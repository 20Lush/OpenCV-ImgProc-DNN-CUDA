#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/dnn/all_layers.hpp>

using namespace std;
using namespace cv;
using namespace dnn;

int main(int, char**) {

    string BASE_PATH = "C:/Users/zachf/Documents/.Computer Vision/opencv-gpu-test/";
    vector<string> CLASS_NAMES;
    ifstream ifs(string(BASE_PATH + "coco.names").c_str());
    string line;

    while(getline(ifs, line)){
        cout << line << endl;
        CLASS_NAMES.push_back(line);
    }

    auto net = readNetFromDarknet(BASE_PATH + "yolov3.cfg", BASE_PATH + "yolov3.weights");

    net.setPreferableBackend(DNN_BACKEND_CUDA);
    net.setPreferableTarget(DNN_TARGET_CUDA);

    Mat image = imread(BASE_PATH + "horse.jpg");

    Mat blob = blobFromImage(image, 1/255, Size(416,416), true, false);
    string ln;
    net.setInput(blob);
    auto outputs = net.forward(ln);


    cout << ln << endl;

    imshow("image", image);

    waitKey(0);
    destroyAllWindows();
    return 0;

}
