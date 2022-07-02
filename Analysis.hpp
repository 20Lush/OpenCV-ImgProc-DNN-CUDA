#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/dnn/all_layers.hpp>
#include <opencv2/cudaimgproc.hpp>

using namespace std;
using namespace cv;
using namespace dnn;

// 7/1/2022 --------------------------
// -> need to program the closest-detection algo
// -----. store the index of the closest detection, run dist as pythagorean theorem btw center and box's center.

class Analysis {
    public:
        const float CONFIDENCE_THRESHOLD = 0.5f; //should probably have this initialized somewhere else for customizeability
        const float NMS_THRESHOLD = 0.3f;
        const int INPUT_WIDTH = 416; // in pixels
        const int INPUT_HEIGHT = 416; // in pixels
        const int IMG_WIDTH = 2560;
        const int IMG_HEIGHT = 1440;
        int DET_COUNT;
        Point closest, current;

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

                        int centerX = (int)(data_ptr[0] * frame.cols); // detection's x val * frame's col pixel count
                        int centerY = (int)(data_ptr[1] * frame.rows); // detection's y val * frame's row pixel count
                        int width = (int)(data_ptr[2] * frame.cols);  // 
                        int height = (int)(data_ptr[3] * frame.rows); 
                        int left = (int)(centerX - (width * 0.5));
                        int top = (int)(centerY - (height * 0.5));

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
                drawCorrectionVector(frame, box);
                DET_COUNT++;
            }

        }

        void drawDetectionCount(Mat& frame){

            String count = to_string(DET_COUNT);
            int drawline;
            Size textSize = getTextSize(count, FONT_HERSHEY_SIMPLEX, 0.75, 1, &drawline);
            //top = max(top, textSize.height);
            rectangle(frame, Point(0,5), Point((int)(round(1.75*textSize.width)), (int)(round(1.75*textSize.height))),  Scalar(255,255,255), FILLED);

            putText(frame, count, Point(5,25), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0,0,0),1);
        }

    private:
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
            return Point( (int)( (box.x + (box.width*0.5)) ), (int)( (box.y + (box.height * 0.5)) ) ); 
        }

        void drawCenterDot(Mat& image, Rect box){
            Point m_center = rectangleCenter(box);
            circle(image, rectangleCenter(box), 5, Scalar(50,178,255), FILLED);
        }

        void drawCorrectionVector(Mat& image, Rect box){
            Point detection_center = rectangleCenter(box);
            Point pov_center = ((int)(INPUT_WIDTH*0.5), (int)(INPUT_HEIGHT*0.5));
            line(image, detection_center, pov_center, Scalar(50,178,255));
            int dx = detection_center.x - pov_center.x; //for reference
            int dy = detection_center.y - pov_center.y;
        }

};

