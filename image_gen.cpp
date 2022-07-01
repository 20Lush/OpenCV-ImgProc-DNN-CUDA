#include "image_gen.hpp"
#include "Analysis.hpp"

int main(){

    //create folder
    //get frame
    //frame crop (gpu method?) to 416x416 centered
    //deposit every 5th(variable) frame
    //keep track of how many frames grabbed, decouple stream once count = 1000
    //repeat until ~8 or so folders of 1k images

    //for early dev reasons, compute only 1 unit and go through it to make sure desired behavior is achieved

    //-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    // -> have a dedicated pool ran on older model with low confidence threshold? what ratio?
    // -----. this is getting added. It would be advantageous to use the other models to grab known good data to reinforce certain stimuli (like partially obscured object association).
    // -----. hopefully this will yield a fairly substantial amount of difficult contexts that the net can be trained on, though needs to be regulated some how.
    // -----. need to avoid data-set lacking edge cases AS WELL AS avoid the data-set becoming too oversaturated with edge cases.
    // -----. though this addition would certainly train out some deviancy in the previous model (i.e. seeing player dead bodies and briefly assigning them a detection).
    //
    // -> image analysis to grab 1 true negative for every 10 known positive? (look up the reccomended ratio).
    // -----. could also be pre-screened with the previous model or previous iter. the problem is that some screened-true-negatives might actually just be a non-detection case from the og model.
    // -----. so a non-zero portion of the presumed true-negatives would get eaten by previous model's innaccuracy.
    //-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


}