#ifndef POLY_LAYER_H
#define POLY_LAYER_H

#include "layer.h"
#include "network.h"

#ifdef __cplusplus
extern "C" {
#endif
layer make_poly_layer(int batch, int w, int h, int n, int total, int *mask, int classes, int max_boxes, int angle_step);
void  forward_poly_layer(const layer l, network_state state);
void  backward_poly_layer(const layer l, network_state state);
void  resize_poly_layer(layer *l, int w, int h);
int   poly_num_detections(layer l, float thresh);
int   poly_num_detections_batch(layer l, float thresh, int batch);
int   get_poly_detections(layer l, int w, int h, int netw, int neth, float thresh, int *map, int relative, detection *dets, int letter);
int   get_poly_detections_batch(layer l, int w, int h, int netw, int neth, float thresh, int *map, int relative, detection *dets, int letter, int batch);
void  correct_poly_boxes(detection *dets, int n, int w, int h, int netw, int neth, int relative, int letter, int poly_angles);

#ifdef GPU
void  forward_poly_layer_gpu(const layer l, network_state state);
void  backward_poly_layer_gpu(layer l, network_state state);
#endif

#ifdef __cplusplus
}
#endif
#endif
