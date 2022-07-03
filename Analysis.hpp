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

        int INPUT_WIDTH = 416;
        int INPUT_HEIGHT = 416;
        int IMG_WIDTH = 2560;
        int IMG_HEIGHT = 1440;

        int DET_COUNT, closest_idx; //make sure to always reset DET_COUNT to 0 on every cycle
        Point closest, current;
        Point center{ (int)(INPUT_WIDTH*0.5), (int)(INPUT_HEIGHT*0.5)};

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

            for(int i = 0; i < outputs.size(); i++){ // taking each output from the net and doing a cumbersome O(n^2) walk over them to refine them to an intuitive format

                float* data_ptr = (float*)outputs[i].data; // points to a row(detection w/ associated info)

                for(int j = 0; j < outputs[i].rows; j++, data_ptr += outputs[i].cols){ // walk through the row at a given col

                    Mat scores = outputs[i].row(j).colRange(5, outputs[i].cols); // locates class confidence values for each detection matrix (all values are > cols[5])
                    Point classIDPoint;                                          // to explain better: col[0->4] are all static properties of the detection like screen coordinates ect
                    double confidence;

                    minMaxLoc(scores, 0, &confidence, 0, &classIDPoint); // finds maximum confidence core with associated class ID in detection matrix

                    if(confidence > CONFIDENCE_THRESHOLD){ // extrapolating useful properties from each net detection

                        int centerX = (int)(data_ptr[0] * frame.cols); // detection's x val * frame's col pixel count
                        int centerY = (int)(data_ptr[1] * frame.rows); // detection's y val * frame's row pixel count
                        int width = (int)(data_ptr[2] * frame.cols);
                        int height = (int)(data_ptr[3] * frame.rows); 
                        int left = (int)(centerX - (width * 0.5)); //a very mid and unsatisfying way to find these values
                        int top = (int)(centerY - (height * 0.5));

                        //the goal outputs of this block.
                        class_IDs.push_back(classIDPoint.x);
                        confidences.push_back((float) confidence);
                        boxes.push_back(Rect(left, top, width, height));

                    }
                }
            }

            //true post-processing where non-max supression eliminates overlaps, modulated with NMS_THRESHOLD
            vector<int> indices;
            NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD, NMS_THRESHOLD, indices); //suppress non-max conf value bounding boxes and eliminate overlapping boxes

            closest = center; //center of FOV

            // iter through the detections and do what you need to do to them. current det index is idx
            for(size_t i = 0; i < indices.size(); i++){
                int idx = indices[i];
                Rect box = boxes[idx];

                current = rectangleCenter(box);
                if(findAbsoluteDistanceFromCenter(current) < findAbsoluteDistanceFromCenter(closest)){
                    closest = current;
                    closest_idx = idx; 
                }

                //compute findAbsoluteDistanceFromCenter into mouse moves and send through serial

                drawBoundingBox(class_IDs[idx], confidences[idx], box.x, box.y, box.x + box.width, box.y + box.height, frame, classes);

                drawCenterDot(frame, boxes[closest_idx]);
                drawCorrectionVector(frame, boxes[closest_idx]);

                DET_COUNT++;
            }

        }

        void drawDetectionCount(Mat& frame){ //Draws a counter on the top left corner of the image

            String count = to_string(DET_COUNT);
            int drawline;
            Size textSize = getTextSize(count, FONT_HERSHEY_SIMPLEX, 0.75, 1, &drawline);
            //top = max(top, textSize.height);
            rectangle(frame, Point(0,5), Point((int)(round(1.75*textSize.width)), (int)(round(1.75*textSize.height))),  Scalar(255,255,255), FILLED);

            putText(frame, count, Point(5,25), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0,0,0),1);
        }

    private:
        Point rectangleCenter(Rect box){ //generic center [might evolve to upper centroid later]
            return Point( (int)( (box.x + (box.width*0.5)) ), (int)( (box.y + (box.height * 0.5)) ) ); 
        }

        double findAbsoluteDistanceFromCenter(Point pt){ //find the absolute distance from pt1 to pt2
            return double(sqrt( pow((pt.x - center.x),2) + pow((pt.y - center.y),2) ));
        }
        void drawBoundingBox(int classID, float confidence, int left, int top, int right, int bottom, Mat& frame, vector<string>& classes){

            // draws a rectangle using the extrapolated coords from postProcess
            rectangle(frame, Point(left,top), Point(right, bottom), Scalar(255,178,50), 3);

            // literal black magic to get text to display nicely with a filled backdrop
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

        void drawCenterDot(Mat& image, Rect box){
            Point m_center = rectangleCenter(box);
            circle(image, rectangleCenter(box), 5, Scalar(50,178,255), FILLED);
        }

        void drawCorrectionVector(Mat& image, Rect box){
            Point detection_center = rectangleCenter(box);
            line(image, detection_center, center, Scalar(50,178,255));

        }

};

