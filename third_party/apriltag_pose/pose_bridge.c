/* EmbodiedVG bridge. The pose algorithm is unmodified AprilRobotics code. */
#include <math.h>
#include <string.h>
#include "apriltag_pose.h"

int embodiedvg_estimate_tag_pose(const double *homography, const double *corners,
                               double tagsize, double fx, double fy, double cx, double cy,
                               double *rotation, double *translation, double *error)
{
    apriltag_detection_t detection = {0};
    detection.H = matd_create_data(3, 3, homography);
    memcpy(detection.p, corners, sizeof(detection.p));
    apriltag_detection_info_t info = {&detection, tagsize, fx, fy, cx, cy};
    apriltag_pose_t pose = {0};
    *error = estimate_tag_pose(&info, &pose);
    int valid = pose.R && pose.t && isfinite(*error);
    if (valid) {
        memcpy(rotation, pose.R->data, 9 * sizeof(double));
        memcpy(translation, pose.t->data, 3 * sizeof(double));
    }
    matd_destroy(pose.R);
    matd_destroy(pose.t);
    matd_destroy(detection.H);
    return valid ? 0 : -1;
}
