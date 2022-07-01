#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/dnn/all_layers.hpp>
#include <opencv2/cudaimgproc.hpp>

constexpr float CONFIDENCE_THRESHOLD = 0.4f;
constexpr float NMS_THRESHOLD = 0.3f;
constexpr int INPUT_WIDTH = 416; // in pixels
constexpr int INPUT_HEIGHT = 416; // in pixels

using namespace std;
using namespace cv;
using namespace dnn;

vector<String> getOutputNames(const Net& net){

    static vector<String> names;
    if(names.empty()){

        vector<int> output_layers = net.getUnconnectedOutLayers(); // gets indices of output layers -> layers without unconnected outputs
        vector<String> layers_names = net.getLayerNames(); // gets names of all layers in net

        names.resize(output_layers.size()); // compile all found names of output layers into string vector
        for(int i = 0; i < output_layers.size(); i++)
            names[i] = layers_names[output_layers[i] - 1];

    }
    
    return names;
}

Rect getCenterSquare(int screen_width, int screen_height, int length){ //returns a Rect equivalent to the (x,y) & (w,h) coords representing the center of the screen. Use to crop out everything but center.

    int sqr_offset = (int)(length * 0.5); //at center coord, add to (Y) to get opencv rect y coord. subtract from (X) to get opencv rect x coord.
    int sqr_centerX = (int)(screen_width * 0.5);
    int sqr_centerY = (int)(screen_height * 0.5);
    
    return Rect(sqr_centerX - sqr_offset, sqr_centerY - sqr_offset, length, length);

}

void drawBoundingBox(int classID, float confidence, int left, int top, int right, int bottom, Mat& frame, vector<string>& classes){

    rectangle(frame, Point(left,top), Point(right, bottom), Scalar(255,178,50), 3);

    string label = format("%.2f", confidence);

    if(!classes.empty()){

        CV_Assert( classID < (int)classes.size());
        label = classes[classID] + ":" + label;

    }
    
    int drawline;
    Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &drawline);
    top = max(top, labelSize.height);
    rectangle(frame, Point(left,(int)(top - round(1.5*labelSize.height))), Point((int)(left+round(1.5*labelSize.width)), top+drawline),  Scalar(255,255,255), FILLED);

    putText(frame, label, Point(left, top), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0,0,0), 1);


}

Point rectangleCenter(Rect box){ //generic center [might evolve to upper centroid later]
    return Point( (0.5 * box.width), (0.5 * box.height) ); 
}

void drawCenterDot(Mat& image, Rect box){
    Point m_center = rectangleCenter(box);
    circle(image, rectangleCenter(box), 5, Scalar(50,178,255), FILLED);
}

void postProcess(Mat& frame, const vector<Mat>& outputs, vector<string>& classes){

    vector<int> class_IDs;
    vector<float> confidences;
    vector<Rect> boxes;

    for(int i = 0; i < outputs.size(); i++){

        float* data_ptr = (float*)outputs[i].data; // points to a row(detection w/ associated info)

        for(int j = 0; j < outputs[i].rows; j++, data_ptr += outputs[i].cols){

            Mat scores = outputs[i].row(j).colRange(5, outputs[i].cols); // locates class confidence values for each detection matrix (all values are > cols[5])
            Point classIDPoint;
            double confidence;

            minMaxLoc(scores, 0, &confidence, 0, &classIDPoint); // finds maximum confidence core with associated class ID in detection matrix

            if(confidence > CONFIDENCE_THRESHOLD){

                int centerX = (int)(data_ptr[0] * frame.cols); // detections x val * frame's col pixel count
                int centerY = (int)(data_ptr[1] * frame.rows); // detections y val * frame's row pixel count
                int width = (int)(data_ptr[2] * frame.cols);  // 
                int height = (int)(data_ptr[3] * frame.rows); 
                int left = centerX - (width/2);
                int top = centerY - (height/2);

                class_IDs.push_back(classIDPoint.x);
                confidences.push_back((float) confidence);
                boxes.push_back(Rect(left, top, width, height));

            }

        }

    }

    //suppress non-max conf value bounding boxes and eliminate overlapping boxes

    vector<int> indices;
    NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD, NMS_THRESHOLD, indices);
    for(size_t i = 0; i < indices.size(); i++){
        int idx = indices[i];
        Rect box = boxes[idx];
        drawBoundingBox(class_IDs[idx], confidences[idx], box.x, box.y, box.x + box.width, box.y + box.height, frame, classes);
        drawCenterDot(frame, box);
    }

}

