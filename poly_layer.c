#include "poly_layer.h"
#include "activations.h"
#include "blas.h"
#include "box.h"
#include "dark_cuda.h"
#include "utils.h"

#include <math.h>
#include <stdio.h>
#include <assert.h>
#include <string.h>
#include <stdlib.h>

extern int check_mistakes;

layer make_poly_layer(int batch, int w, int h, int n, int total, int *mask, int classes, int max_boxes, int angle_step)
{
    int i;
    layer l = { (LAYER_TYPE)0 };
    l.type = POLY_YOLO;
	l.poly_angle_step = angle_step;
	l.poly_angles = (int)(360/angle_step);
    l.n = n;
    l.total = total;
    l.batch = batch;
    l.h = h;
    l.w = w;
    l.c = n*(classes + 4 + 1 + l.poly_angles*3);
    l.out_w = l.w;
    l.out_h = l.h;
    l.out_c = l.c;
    l.classes = classes;
    l.cost = (float*)xcalloc(1, sizeof(float));
    l.biases = (float*)xcalloc(total*2, sizeof(float));
    if(mask) l.mask = mask;
    else{
        l.mask = (int*)xcalloc(n, sizeof(int));
        for(i = 0; i < n; ++i){
            l.mask[i] = i;
        }
    }
    l.bias_updates = (float*)xcalloc(n*2, sizeof(float));
    l.outputs = h*w*n*(classes + 4 + 1 + l.poly_angles*3);
    l.inputs = l.outputs;
    l.max_boxes = max_boxes;
    l.truths = l.max_boxes*(4 + 1 + POLY_MAX_VERTICES*2);    // 90*(4 + 1 + POLY_MAX_VERTICES_NUM*2); 
    l.delta = (float*)xcalloc(batch*l.outputs, sizeof(float));
    l.output = (float*)xcalloc(batch*l.outputs, sizeof(float));
    for(i = 0; i < total*2; ++i){
        l.biases[i] = .5;
    }
    l.forward = forward_poly_layer;
    l.backward = backward_poly_layer;
#ifdef GPU
    l.forward_gpu = forward_poly_layer_gpu;
    l.backward_gpu = backward_poly_layer_gpu;
    l.output_gpu = cuda_make_array(l.output, batch*l.outputs);
    l.output_avg_gpu = cuda_make_array(l.output, batch*l.outputs);
    l.delta_gpu = cuda_make_array(l.delta, batch*l.outputs);

    free(l.output);
    if (cudaSuccess == cudaHostAlloc(&l.output, batch*l.outputs*sizeof(float), cudaHostRegisterMapped)) l.output_pinned = 1;
    else {
        cudaGetLastError(); // reset CUDA-error
        l.output = (float*)xcalloc(batch*l.outputs, sizeof(float));
    }
    free(l.delta);
    if (cudaSuccess == cudaHostAlloc(&l.delta, batch*l.outputs*sizeof(float), cudaHostRegisterMapped)) l.delta_pinned = 1;
    else {
        cudaGetLastError(); // reset CUDA-error
        l.delta = (float*)xcalloc(batch*l.outputs, sizeof(float));
    }
#endif
    fprintf(stderr, "poly-yolo\n");
    srand(time(0));

    return l;
}

void resize_poly_layer(layer *l, int w, int h)
{
    l->w = w;
    l->h = h;
    l->outputs = h*w*l->n*(l->classes + 4 + 1 + l->poly_angles*3);
    l->inputs = l->outputs;

    if (!l->output_pinned) l->output = (float*)xrealloc(l->output, l->batch*l->outputs*sizeof(float));
    if (!l->delta_pinned) l->delta = (float*)xrealloc(l->delta, l->batch*l->outputs*sizeof(float));

#ifdef GPU
    if (l->output_pinned) {
        CHECK_CUDA(cudaFreeHost(l->output));
        if (cudaSuccess != cudaHostAlloc(&l->output, l->batch*l->outputs*sizeof(float), cudaHostRegisterMapped)) {
            cudaGetLastError(); // reset CUDA-error
            l->output = (float*)xcalloc(l->batch*l->outputs, sizeof(float));
            l->output_pinned = 0;
        }
    }

    if (l->delta_pinned) {
        CHECK_CUDA(cudaFreeHost(l->delta));
        if (cudaSuccess != cudaHostAlloc(&l->delta, l->batch*l->outputs*sizeof(float), cudaHostRegisterMapped)) {
            cudaGetLastError(); // reset CUDA-error
            l->delta = (float*)xcalloc(l->batch*l->outputs, sizeof(float));
            l->delta_pinned = 0;
        }
    }

    cuda_free(l->delta_gpu);
    cuda_free(l->output_gpu);
    cuda_free(l->output_avg_gpu);

    l->delta_gpu = cuda_make_array(l->delta, l->batch*l->outputs);
    l->output_gpu = cuda_make_array(l->output, l->batch*l->outputs);
    l->output_avg_gpu = cuda_make_array(l->output, l->batch*l->outputs);
#endif
}

poly_box get_poly_box(float *x, float *biases, int n, int index, int i, int j, int lw, int lh, int w, int h, int stride, int classes, int poly_angles, int poly_angle_step, int flag)
{
	int k;
	poly_box b;
	float biase_rlo, dist, diagonal;

    b.x = (i + x[index + 0*stride])/lw;
    b.y = (j + x[index + 1*stride])/lh;
    b.w = exp(x[index + 2*stride])*biases[2*n]/w;
    b.h = exp(x[index + 3*stride])*biases[2*n+1]/h;
	
	if (flag == 1) {
		diagonal = sqrtf(b.w*b.w*w*w + b.h*b.h*h*h);
		biase_rlo = sqrtf(biases[2*n]*biases[2*n] + biases[2*n+1]*biases[2*n+1])/2;
		dist = sqrtf(w*w + h*h);

		for (k = 0; k < poly_angles; ++k) {

			float rlo = exp(x[index + (5 + classes + k*3 + 0)*stride])*biase_rlo/dist;
			float alpha = (x[index + (5 + classes + k*3 + 1)*stride]*poly_angle_step + k*poly_angle_step)/180*M_PI;
			float prob = x[index + (5 + classes + k*3 + 2)*stride];

			//printf("k = %d, rlo = %0.4f%, alpha = %0.4f, prob = %0.4f\n", k, rlo, alpha, prob);
		
			b.prs[k].x = (rlo*diagonal)*cosf(alpha)/w + b.x;
			b.prs[k].y = (rlo*diagonal)*sinf(alpha)/h + b.y;
			b.prs[k].prob = prob;
		}
	}

    return b;
}

static inline float fix_nan_inf(float val)
{
    if (isnan(val) || isinf(val)) val = 0;
    return val;
}

static inline float clip_value(float val, const float max_val)
{
    if (val > max_val) {
        val = max_val;
    } else if (val < -max_val) {
        val = -max_val;
    }
    return val;
}

void delta_poly_polygon(poly_box truth, float *x, float *biases, int n, int index, int w, int h, float *delta, float scale, int stride, float poly_rlo_normalizer, float poly_alpha_normalizer, int poly_angles)
{
	int k;
	float t_rlo, t_alpha, t_prob;
	float biases_rlo, dist;

	biases_rlo = sqrtf(biases[2*n]*biases[2*n] + biases[2*n + 1]*biases[2*n + 1])/2;
	dist = sqrtf(w*w + h*h);

	for (k = 0; k < poly_angles; ++k) {

		if (truth.prs[k].prob == 1) {			
			t_rlo = fix_nan_inf(log(truth.prs[k].rlo*dist/biases_rlo));
			t_alpha = truth.prs[k].alpha;
			delta[index + (k*3 + 0)*stride] += scale*(t_rlo - x[index + (k*3 + 0)*stride])*poly_rlo_normalizer;
			delta[index + (k*3 + 1)*stride] += scale*(t_alpha - x[index + (k*3 + 1)*stride])*poly_alpha_normalizer;
			//delta[index + (k*3 + 2)*stride] += (1 - x[index + (k*3 + 2)*stride])*poly_prob_normalizer;
		} 
	}	

	//getchar();

}

ious delta_poly_box(poly_box truth, float *x, float *biases, int n, int index, int i, int j, int lw, int lh, int w, int h, float *delta, float scale, int stride, float iou_normalizer, IOU_LOSS iou_loss, int accumulate, float max_delta,int classes, int poly_angles, int poly_angle_step)
{
    ious all_ious = { 0 };
    //i - step in layer width
    //j - step in layer height
    //returns a box in absolute coordinates
    poly_box pred = get_poly_box(x, biases, n, index, i, j, lw, lh, w, h, stride, classes, poly_angles, poly_angle_step, 0);
    all_ious.iou = poly_box_iou(pred, truth);
    all_ious.giou = poly_box_giou(pred, truth);
    all_ious.diou = poly_box_diou(pred, truth);
    all_ious.ciou = poly_box_ciou(pred, truth);
    
	//avoid nan in dx_box_iou
    if (pred.w == 0) { pred.w = 1.0; }
    if (pred.h == 0) { pred.h = 1.0; }
    if (iou_loss == MSE) {   // old loss
        float tx = (truth.x*lw - i);
        float ty = (truth.y*lh - j);
        float tw = log(truth.w*w/biases[2*n]);
        float th = log(truth.h*h/biases[2*n + 1]);
        //printf(" tx = %f, ty = %f, tw = %f, th = %f \n", tx, ty, tw, th);
        //printf(" x = %f, y = %f, w = %f, h = %f \n", x[index + 0*stride], x[index + 1*stride], x[index + 2*stride], x[index + 3*stride]);
        //accumulate delta
        delta[index + 0*stride] += scale * (tx - x[index + 0*stride]) * iou_normalizer;
        delta[index + 1*stride] += scale * (ty - x[index + 1*stride]) * iou_normalizer;
        delta[index + 2*stride] += scale * (tw - x[index + 2*stride]) * iou_normalizer;
        delta[index + 3*stride] += scale * (th - x[index + 3*stride]) * iou_normalizer;
    } else {
        //https://github.com/generalized-iou/g-darknet
        //https://arxiv.org/abs/1902.09630v2
        //https://giou.stanford.edu/
        all_ious.dx_iou = dx_poly_box_iou(pred, truth, iou_loss);

        //jacobian^t (transpose)
        //float dx = (all_ious.dx_iou.dl + all_ious.dx_iou.dr);
        //float dy = (all_ious.dx_iou.dt + all_ious.dx_iou.db);
        //float dw = ((-0.5*all_ious.dx_iou.dl) + (0.5*all_ious.dx_iou.dr));
        //float dh = ((-0.5*all_ious.dx_iou.dt) + (0.5*all_ious.dx_iou.db));

        //jacobian^t (transpose)
        float dx = all_ious.dx_iou.dt;
        float dy = all_ious.dx_iou.db;
        float dw = all_ious.dx_iou.dl;
        float dh = all_ious.dx_iou.dr;

        //predict exponential, apply gradient of e^delta_t ONLY for w,h
        dw *= exp(x[index + 2*stride]);
        dh *= exp(x[index + 3*stride]);

        //normalize iou weight
        dx *= iou_normalizer;
        dy *= iou_normalizer;
        dw *= iou_normalizer;
        dh *= iou_normalizer;

        dx = fix_nan_inf(dx);
        dy = fix_nan_inf(dy);
        dw = fix_nan_inf(dw);
        dh = fix_nan_inf(dh);

        if (max_delta != FLT_MAX) {
            dx = clip_value(dx, max_delta);
            dy = clip_value(dy, max_delta);
            dw = clip_value(dw, max_delta);
            dh = clip_value(dh, max_delta);
        }
        if (!accumulate) {
            delta[index + 0*stride] = 0;
            delta[index + 1*stride] = 0;
            delta[index + 2*stride] = 0;
            delta[index + 3*stride] = 0;
        }
        //accumulate delta
        delta[index + 0*stride] += dx;
        delta[index + 1*stride] += dy;
        delta[index + 2*stride] += dw;
        delta[index + 3*stride] += dh;
    }
    return all_ious;
}


void averages_poly_deltas(int class_index, int box_index, int stride, int classes, float *delta)
{
    int k,c,classes_in_one_box = 0;
    for (c = 0; c < classes; ++c) {
        if (delta[class_index + stride*c] > 0) classes_in_one_box++;
    }
    if (classes_in_one_box > 0) {
        delta[box_index + 0*stride] /= classes_in_one_box;
        delta[box_index + 1*stride] /= classes_in_one_box;
        delta[box_index + 2*stride] /= classes_in_one_box;
        delta[box_index + 3*stride] /= classes_in_one_box;
		//for (k = 0; k < l.poly_angles; ++k) {
		//	delta[box_index + (5 + classes + k*3 + 0)*stride] /= classes_in_one_box;
		//	delta[box_index + (5 + classes + k*3 + 1)*stride] /= classes_in_one_box;
		//  delta[box_index + (5 + classes + k*3 + 2)*stride] /= classes_in_one_box;
		//}
    }
}

void delta_poly_class(float *output, float *delta, int index, int class_id, int classes, int stride, float *avg_cat, int focal_loss, float label_smooth_eps, float *classes_multipliers)
{
    int n;
    if (delta[index + stride*class_id]){
        float y_true = 1;
        if(label_smooth_eps) y_true = y_true *  (1 - label_smooth_eps) + 0.5*label_smooth_eps;
        float result_delta = y_true - output[index + stride*class_id];
        if(!isnan(result_delta) && !isinf(result_delta)) delta[index + stride*class_id] = result_delta;
        //delta[index + stride*class_id] = 1 - output[index + stride*class_id];
        if (classes_multipliers) delta[index + stride*class_id] *= classes_multipliers[class_id];
        if(avg_cat) *avg_cat += output[index + stride*class_id];
        return;
    }
    // focal loss
    if (focal_loss) {
        // focal Loss
        float alpha = 0.5;    // 0.25 or 0.5
        //float gamma = 2;    // hardcoded in many places of the grad-formula

        int ti = index + stride*class_id;
        float pt = output[ti] + 0.000000000000001F;
        // http://fooplot.com/#W3sidHlwZSI6MCwiZXEiOiItKDEteCkqKDIqeCpsb2coeCkreC0xKSIsImNvbG9yIjoiIzAwMDAwMCJ9LHsidHlwZSI6MTAwMH1d
        float grad = -(1 - pt)*(2*pt*logf(pt) + pt - 1);    // http://blog.csdn.net/linmingan/article/details/77885832
        //float grad = (1 - pt)*(2*pt*logf(pt) + pt - 1);    // https://github.com/unsky/focal-loss

        for (n = 0; n < classes; ++n) {
            delta[index + stride*n] = (((n == class_id) ? 1 : 0) - output[index + stride*n]);
            delta[index + stride*n] *= alpha*grad;
            if (n == class_id && avg_cat) *avg_cat += output[index + stride*n];
        }
    }
    else {
        // default
        for (n = 0; n < classes; ++n) {
            float y_true = ((n == class_id) ? 1 : 0);
            if (label_smooth_eps) y_true = y_true*(1 - label_smooth_eps) + 0.5*label_smooth_eps;
            float result_delta = y_true - output[index + stride*n];
            if (!isnan(result_delta) && !isinf(result_delta)) delta[index + stride*n] = result_delta;
            if (classes_multipliers && n == class_id) delta[index + stride*class_id] *= classes_multipliers[class_id];
            if (n == class_id && avg_cat) *avg_cat += output[index + stride*n];
        }
    }
}

int compare_poly_class(float *output, int classes, int class_index, int stride, float objectness, int class_id, float conf_thresh)
{
    int j;
    for (j = 0; j < classes; ++j) {
        //float prob = objectness*output[class_index + stride*j];
        float prob = output[class_index + stride*j];
        if (prob > conf_thresh) {
            return 1;
        }
    }
    return 0;
}

static int entry_poly_index(layer l, int batch, int location, int entry)
{
    int n = location/(l.w*l.h);
    int loc = location%(l.w*l.h);
    return (batch*l.outputs + n*l.w*l.h*(4 + 1 + l.classes + l.poly_angles*3) + entry*l.w*l.h + loc);
}

void forward_poly_layer(const layer l, network_state state)
{
    int i, j, b, t, n, k, index;
    memcpy(l.output, state.input, l.outputs*l.batch*sizeof(float));
#ifndef GPU
    for (b = 0; b < l.batch; ++b) {
        for (n = 0; n < l.n; ++n) {
            index = entry_poly_index(l, b, n*l.w*l.h, 0);
            activate_array(l.output + index, 2*l.w*l.h, LOGISTIC);        // x,y,
            scal_add_cpu(2*l.w*l.h, l.scale_x_y, -0.5*(l.scale_x_y - 1), l.output + index, 1);    // scale x,y
            index = entry_poly_index(l, b, n*l.w*l.h, 4);
            activate_array(l.output + index, (1 + l.classes)*l.w*l.h, LOGISTIC);
			for (k = 0; k < l.poly_angles; ++k) {
				index = entry_poly_index(l, b, n*l.w*l.h, 4 + 1 + l.classes + k*3 + 1);
				activate_array(l.output + index, 2*l.w*l.h, LOGISTIC);
			}
        }
    }
#endif
    // delta is zeroed
    memset(l.delta, 0, l.outputs*l.batch*sizeof(float));
    if (!state.train) return;
    //float avg_iou = 0;
    float tot_iou = 0;
    float tot_giou = 0;
    float tot_diou = 0;
    float tot_ciou = 0;
    float tot_iou_loss = 0;
    float tot_giou_loss = 0;
    float tot_diou_loss = 0;
    float tot_ciou_loss = 0;
    float recall = 0;
    float recall75 = 0;
    float avg_cat = 0;
    float avg_obj = 0;
    float avg_anyobj = 0;
	float avg_poly_rlo = 0;
	float avg_poly_alpha = 0;
	float avg_poly_prob = 0;
    int count = 0;
    int class_count = 0;
	int poly_count = 0;
    *(l.cost) = 0;
    for (b = 0; b < l.batch; ++b) {
        for (j = 0; j < l.h; ++j) {
            for (i = 0; i < l.w; ++i) {
                for (n = 0; n < l.n; ++n) {
                    const int cls_index = entry_poly_index(l, b, n*l.w*l.h + j*l.w + i, 4 + 1);
                    const int obj_index = entry_poly_index(l, b, n*l.w*l.h + j*l.w + i, 4);
                    const int box_index = entry_poly_index(l, b, n*l.w*l.h + j*l.w + i, 0);
					const int poly_index = entry_poly_index(l, b, n*l.w*l.h + j*l.w + i, 4 + 1 + l.classes);
                    const int stride = l.w*l.h;
                    poly_box pred = get_poly_box(l.output, l.biases, l.mask[n], box_index, i, j, l.w, l.h, state.net.w, state.net.h, l.w*l.h, l.classes, l.poly_angles, l.poly_angle_step, 0);
                    float best_match_iou = 0;
                    int best_match_t = 0;
                    float best_iou = 0;
                    int best_t = 0;
					poly_box best_match_poly_box;
					poly_box best_iou_poly_box;
                    for (t = 0; t < l.max_boxes; ++t) {

                        poly_box truth = float_to_poly_box_stride(state.truth + t*(4 + 1 + POLY_MAX_VERTICES*2) + b*l.truths, 1, l.poly_angles, l.poly_angle_step, state.net.w, state.net.h);
						int class_id = state.truth[t*(4 + 1 + POLY_MAX_VERTICES*2) + b*l.truths + 4];

                        if (class_id>= l.classes || class_id < 0) {
                            printf("\n Warning: in txt-labels class_id=%d >= classes=%d in cfg-file. In txt-labels class_id should be [from 0 to %d] \n", class_id, l.classes, l.classes - 1);
                            printf("\n truth.x = %f, truth.y = %f, truth.w = %f, truth.h = %f, class_id = %d \n", truth.x, truth.y, truth.w, truth.h, class_id);
                            if (check_mistakes) getchar();
                            continue; // if label contains class_id more than number of classes in the cfg-file and class_id check garbage value
                        }
                        if (!truth.x) break;  // continue;

                        float objectness = l.output[obj_index];
						l.output[obj_index] = fix_nan_inf(l.output[obj_index]);
						for (k = 0; k < l.poly_angles; ++k) {
							int poly_prob_index = poly_index + (k*3 + 2)*l.w*l.h;
							l.output[poly_prob_index] = fix_nan_inf(l.output[poly_prob_index]);
						}
                        int class_id_match = compare_poly_class(l.output, l.classes, cls_index, l.w*l.h, objectness, class_id, 0.25f);
                        float iou = poly_box_iou(pred, truth);
                        if (iou > best_match_iou && class_id_match == 1) {
                            best_match_iou = iou;
                            best_match_t = t;
							best_match_poly_box = truth;
                        }
                        if (iou > best_iou) {
                            best_iou = iou;
                            best_t = t;
							best_iou_poly_box = truth;
                        }
                    }

                    avg_anyobj += l.output[obj_index];
                    l.delta[obj_index] = l.cls_normalizer*(0 - l.output[obj_index]);
					for (k = 0; k < l.poly_angles; ++k) {
						l.delta[poly_index + (k*3 + 2)*stride] = l.poly_prob_normalizer*(0 - l.output[poly_index + (k*3 + 2)*stride]);
					}

                    if (best_match_iou > l.ignore_thresh) {//.7
                        const float iou_multiplier = best_match_iou*best_match_iou;// (best_match_iou - l.ignore_thresh) / (1.0 - l.ignore_thresh);
                        if (l.objectness_smooth) {
                            l.delta[obj_index] = l.cls_normalizer*(iou_multiplier - l.output[obj_index]);
							for (k = 0; k < l.poly_angles; ++k) {
								if (best_match_poly_box.prs[k].prob == 1) {
									l.delta[poly_index + (k*3 + 2)*stride] = l.poly_prob_normalizer*(iou_multiplier - l.output[poly_index + (k*3 + 2)*stride]);
								}
							}
                            int class_id = state.truth[best_match_t*(4 + 1 + POLY_MAX_VERTICES*2) + b*l.truths + 4];
                            if (l.map) class_id = l.map[class_id];
                            const float class_multiplier = (l.classes_multipliers) ? l.classes_multipliers[class_id] : 1.0f;
                            l.delta[cls_index + stride*class_id] = class_multiplier*(iou_multiplier - l.output[cls_index + stride*class_id]);
                        } else {
							l.delta[obj_index] = 0;
							for (k = 0; k < l.poly_angles; ++k) {
								if (best_match_poly_box.prs[k].prob == 1) {
									l.delta[poly_index + (k*3 + 2)*stride] = 0;
								}
							}
						}
                    }
                    else if (state.net.adversarial) {
                        float scale = pred.w*pred.h;
                        if (scale > 0) scale = sqrt(scale);
                        l.delta[obj_index] = scale*l.cls_normalizer*(0 - l.output[obj_index]);
						for (k = 0; k < l.poly_angles; ++k) {
							l.delta[poly_index + (k*3 + 2)*stride] = scale*l.poly_prob_normalizer*(0 - l.output[poly_index + (k*3 + 2)*stride]);
						}
                        for (int cl_id = 0; cl_id < l.classes; ++cl_id) {
							if (l.output[cls_index + stride*cl_id] * l.output[obj_index] > 0.25) {
								l.delta[cls_index + stride*cl_id] = scale*(0 - l.output[cls_index + stride*cl_id]);
							}
                        }
                    }

                    if (best_iou > l.truth_thresh) { 

						const float iou_multiplier = best_iou*best_iou;// (best_iou - l.truth_thresh)/(1.0 - l.truth_thresh);
						if (l.objectness_smooth) {
							l.delta[obj_index] = l.cls_normalizer*(iou_multiplier - l.output[obj_index]);
							for (k = 0; k < l.poly_angles; ++k) {
								if (best_match_poly_box.prs[k].prob == 1) {
									l.delta[poly_index + (k*3 + 2)*stride] = l.poly_prob_normalizer*(iou_multiplier - l.output[poly_index + (k*3 + 2)*stride]);
								}
							}
						} else {
							l.delta[obj_index] = l.cls_normalizer*(1 - l.output[obj_index]);
							for (k = 0; k < l.poly_angles; ++k) {
								if (best_match_poly_box.prs[k].prob == 1) {
									l.delta[poly_index + (k*3 + 2)*stride] = l.poly_prob_normalizer*(1 - l.output[poly_index + (k*3 + 2)*stride]);
								}
							}
						}
                        int class_id = state.truth[best_t*(4 + 1 + POLY_MAX_VERTICES*2) + b*l.truths + 4];
                        if (l.map) class_id = l.map[class_id];

                        delta_poly_class(l.output, l.delta, cls_index, class_id, l.classes, l.w*l.h, 0, l.focal_loss, l.label_smooth_eps, l.classes_multipliers);
                        const float class_multiplier = (l.classes_multipliers) ? l.classes_multipliers[class_id] : 1.0f;
                        if (l.objectness_smooth) l.delta[cls_index + stride*class_id] = class_multiplier*(iou_multiplier - l.output[cls_index + stride*class_id]);

                        poly_box truth = float_to_poly_box_stride(state.truth + best_t*(4 + 1 + POLY_MAX_VERTICES*2) + b*l.truths, 1, l.poly_angles, l.poly_angle_step, state.net.w, state.net.h);

                        delta_poly_box(truth, l.output, l.biases, l.mask[n], box_index, i, j, l.w, l.h, state.net.w, state.net.h, l.delta, (2-truth.w*truth.h), l.w*l.h, l.iou_normalizer*class_multiplier, l.iou_loss, 1, l.max_delta, l.classes, l.poly_angles, l.poly_angle_step);
						delta_poly_polygon(truth, l.output, l.biases, l.mask[n], poly_index, state.net.w, state.net.h, l.delta, (2 - truth.w*truth.h), l.w*l.h, l.poly_rlo_normalizer*class_multiplier, l.poly_alpha_normalizer*class_multiplier, l.poly_angles);
                    }

                }
            }
        }
        for (t = 0; t < l.max_boxes; ++t) {

            poly_box truth = float_to_poly_box_stride(state.truth + t*(4 + 1 + POLY_MAX_VERTICES*2) + b*l.truths, 1, l.poly_angles, l.poly_angle_step, state.net.w, state.net.h);

            if (truth.x < 0 || truth.y < 0 || truth.x > 1 || truth.y > 1 || truth.w < 0 || truth.h < 0) {
                char buff[256];
                printf(" Wrong label: truth.x = %f, truth.y = %f, truth.w = %f, truth.h = %f \n", truth.x, truth.y, truth.w, truth.h);
                sprintf(buff, "echo \"Wrong label: truth.x = %f, truth.y = %f, truth.w = %f, truth.h = %f\" >> bad_label.list", truth.x, truth.y, truth.w, truth.h);
                system(buff);
            }

            int class_id = state.truth[t*(4 + 1 + POLY_MAX_VERTICES*2) + b*l.truths + 4];
            if (class_id >= l.classes || class_id < 0) continue; // if label contains class_id more than number of classes in the cfg-file and class_id check garbage value

            if (!truth.x) break;  // continue;
            float best_iou = 0;
            int best_n = 0;
            i = (truth.x*l.w);
            j = (truth.y*l.h);
            poly_box truth_shift = truth;
            truth_shift.x = truth_shift.y = 0;
            for (n = 0; n < l.total; ++n) {
				poly_box pred;
				pred.x = 0;
				pred.y = 0;
                pred.w = l.biases[2*n]/state.net.w;
                pred.h = l.biases[2*n + 1]/state.net.h;
                float iou = poly_box_iou(pred, truth_shift);
                if (iou > best_iou) {
                    best_iou = iou;
                    best_n = n;
                }
            }

            int mask_n = int_index(l.mask, best_n, l.n);
            if (mask_n >= 0) {
                int class_id = state.truth[t*(4 + 1 + POLY_MAX_VERTICES*2) + b*l.truths + 4];
                if (l.map) class_id = l.map[class_id];

                int box_index = entry_poly_index(l, b, mask_n*l.w*l.h + j*l.w + i, 0);
                const float class_multiplier = (l.classes_multipliers) ? l.classes_multipliers[class_id] : 1.0f;
                ious all_ious = delta_poly_box(truth, l.output, l.biases, best_n, box_index, i, j, l.w, l.h, state.net.w, state.net.h, l.delta, (2-truth.w*truth.h), l.w*l.h, l.iou_normalizer*class_multiplier, l.iou_loss, 1, l.max_delta, l.classes, l.poly_angles, l.poly_angle_step);

                // range is 0 <= 1
                tot_iou += all_ious.iou;
                tot_iou_loss += 1 - all_ious.iou;
                // range is -1 <= giou <= 1
                tot_giou += all_ious.giou;
                tot_giou_loss += 1 - all_ious.giou;

                tot_diou += all_ious.diou;
                tot_diou_loss += 1 - all_ious.diou;

                tot_ciou += all_ious.ciou;
                tot_ciou_loss += 1 - all_ious.ciou;

                int obj_index = entry_poly_index(l, b, mask_n*l.w*l.h + j*l.w + i, 4);
                avg_obj += l.output[obj_index];
				//printf("objA=%0.4f\n", l.output[obj_index]);
                l.delta[obj_index] = class_multiplier*l.cls_normalizer*(1 - l.output[obj_index]);
				
				int poly_index = entry_poly_index(l, b, mask_n*l.w*l.h + j*l.w + i, 5+l.classes);
				for (k = 0; k < l.poly_angles; ++k) {
					if (truth.prs[k].prob == 1) {
						l.delta[poly_index + (k*3 + 2)*l.w*l.h] = class_multiplier*l.poly_prob_normalizer*(1 - l.output[poly_index + (k*3 + 2)*l.w*l.h]);
					}
				}

                int class_index = entry_poly_index(l, b, mask_n*l.w*l.h + j*l.w + i, 4 + 1);
                delta_poly_class(l.output, l.delta, class_index, class_id, l.classes, l.w*l.h, &avg_cat, l.focal_loss, l.label_smooth_eps, l.classes_multipliers);

                //printf(" label: class_id = %d, truth.x = %f, truth.y = %f, truth.w = %f, truth.h = %f \n", class_id, truth.x, truth.y, truth.w, truth.h);
                //printf(" mask_n = %d, l.output[obj_index] = %f, l.output[class_index + class_id] = %f \n\n", mask_n, l.output[obj_index], l.output[class_index + class_id]);

                ++count;
                ++class_count;
                if (all_ious.iou > .5) recall += 1;
                if (all_ious.iou > .75) recall75 += 1;

				delta_poly_polygon(truth, l.output, l.biases, best_n, poly_index, state.net.w, state.net.h, l.delta, (2-truth.w*truth.h), l.w*l.h, l.poly_rlo_normalizer*class_multiplier, l.poly_alpha_normalizer*class_multiplier,l.poly_angles);
				
				poly_count++;
#if 1				
				float tmp_avg_poly_rlo = 0;
				float tmp_avg_poly_alpha = 0;
				float tmp_avg_poly_prob = 0;

				int stride = l.w*l.h;

				for (k = 0; k < l.poly_angles; ++k) {
					tmp_avg_poly_rlo += l.output[poly_index + (k*3)*stride];
					tmp_avg_poly_alpha += l.output[poly_index + (k*3 + 1)*stride];
					tmp_avg_poly_prob += l.output[poly_index + (k*3 + 2)*stride];
				}

				tmp_avg_poly_rlo /= l.poly_angles;
				tmp_avg_poly_alpha /= l.poly_angles;
				tmp_avg_poly_prob /= l.poly_angles;

				avg_poly_rlo += tmp_avg_poly_rlo;
				avg_poly_alpha += tmp_avg_poly_alpha;
				avg_poly_prob += tmp_avg_poly_prob;
#endif
            }

            // iou_thresh
            for (n = 0; n < l.total; ++n) {
                int mask_n = int_index(l.mask, n, l.n);
                if (mask_n >= 0 && n != best_n && l.iou_thresh < 1.0f) {
					poly_box pred;
					pred.x = 0;
					pred.y = 0;
                    pred.w = l.biases[2*n]/state.net.w;
                    pred.h = l.biases[2*n + 1]/state.net.h;
                    float iou = poly_box_iou_kind(pred, truth_shift, l.iou_thresh_kind); // IOU, GIOU, MSE, DIOU, CIOU
                    // iou, n
                    if (iou > l.iou_thresh) {				
                        int class_id = state.truth[t*(4 + 1 + POLY_MAX_VERTICES*2) + b*l.truths + 4];
                        if (l.map) class_id = l.map[class_id];

                        int box_index = entry_poly_index(l, b, mask_n*l.w*l.h + j*l.w + i, 0);
                        const float class_multiplier = (l.classes_multipliers) ? l.classes_multipliers[class_id] : 1.0f;
                        ious all_ious = delta_poly_box(truth, l.output, l.biases, n, box_index, i, j, l.w, l.h, state.net.w, state.net.h, l.delta, (2-truth.w*truth.h), l.w*l.h, l.iou_normalizer*class_multiplier, l.iou_loss, 1, l.max_delta, l.classes, l.poly_angles, l.poly_angle_step);

                        // range is 0 <= 1
                        tot_iou += all_ious.iou;
                        tot_iou_loss += 1 - all_ious.iou;
                        // range is -1 <= giou <= 1
                        tot_giou += all_ious.giou;
                        tot_giou_loss += 1 - all_ious.giou;

                        tot_diou += all_ious.diou;
                        tot_diou_loss += 1 - all_ious.diou;

                        tot_ciou += all_ious.ciou;
                        tot_ciou_loss += 1 - all_ious.ciou;

                        int obj_index = entry_poly_index(l, b, mask_n*l.w*l.h + j*l.w + i, 4);
                        avg_obj += l.output[obj_index];
						
                        l.delta[obj_index] = class_multiplier*l.cls_normalizer*(1 - l.output[obj_index]);

						int poly_index = entry_poly_index(l, b, mask_n*l.w*l.h + j*l.w + i, 5+l.classes);
						for (k = 0; k < l.poly_angles; ++k) {
							if (truth.prs[k].prob == 1) {
								l.delta[poly_index + (k*3 + 2)*l.w*l.h] = class_multiplier*l.poly_prob_normalizer*(1 - l.output[poly_index + (k*3 + 2)*l.w*l.h]);
							}
						}
						
                        int class_index = entry_poly_index(l, b, mask_n*l.w*l.h + j*l.w + i, 4 + 1);
                        delta_poly_class(l.output, l.delta, class_index, class_id, l.classes, l.w*l.h, &avg_cat, l.focal_loss, l.label_smooth_eps, l.classes_multipliers);

                        ++count;
                        ++class_count;
                        if (all_ious.iou > .5) recall += 1;
                        if (all_ious.iou > .75) recall75 += 1;
						
						delta_poly_polygon(truth, l.output, l.biases, n, poly_index, state.net.w, state.net.h, l.delta, (2-truth.w*truth.h), l.w*l.h, l.poly_rlo_normalizer*class_multiplier, l.poly_alpha_normalizer*class_multiplier,l.poly_angles);
						
						poly_count++;
#if 1
						float tmp_avg_poly_rlo = 0;
						float tmp_avg_poly_alpha = 0;
						float tmp_avg_poly_prob = 0;

						int stride = l.w*l.h;

						for (k = 0; k < l.poly_angles; ++k) {
							tmp_avg_poly_rlo += l.output[poly_index + (k*3)*stride];
							tmp_avg_poly_alpha += l.output[poly_index + (k*3 + 1)*stride];
							tmp_avg_poly_prob += l.output[poly_index + (k*3 + 2)*stride];
						}

						tmp_avg_poly_rlo /= l.poly_angles;
						tmp_avg_poly_alpha /= l.poly_angles;
						tmp_avg_poly_prob /= l.poly_angles;

						avg_poly_rlo += tmp_avg_poly_rlo;
						avg_poly_alpha += tmp_avg_poly_alpha;
						avg_poly_prob += tmp_avg_poly_prob;
#endif
                    }
                }
            }
        }

        // averages the deltas obtained by the function: delta_yolo_box()_accumulate
        for (j = 0; j < l.h; ++j) {
            for (i = 0; i < l.w; ++i) {
                for (n = 0; n < l.n; ++n) {
                    int box_index = entry_poly_index(l, b, n*l.w*l.h + j*l.w + i, 0);
                    int cls_index = entry_poly_index(l, b, n*l.w*l.h + j*l.w + i, 5);
                    int stride = l.w*l.h;
                    averages_poly_deltas(cls_index, box_index, stride, l.classes, l.delta);
                }
            }
        }

    }

    if (count == 0) count = 1;
    if (class_count == 0) class_count = 1;
	if (poly_count == 0) poly_count = 1;

	int stride = l.w*l.h;
	
#if 0
	//test
	for (b = 0; b < l.batch; ++b) {
		for (j = 0; j < l.h; ++j) {
			for (i = 0; i < l.w; ++i) {
				for (n = 0; n < l.n; ++n) {
					index = entry_poly_index(l, b, n*l.w*l.h + j*l.w + i, 5 + l.classes);
					for (k = 0; k < l.poly_angles; ++k) {
						l.delta[index + (3*k + 0)*stride] = 0;
						l.delta[index + (3*k + 1)*stride] = 0;
						l.delta[index + (3*k + 2)*stride] = 0;
					}
				}
			}
		}
	}
#endif
	//
    float* classification_loss_delta = (float*)calloc(l.batch*l.outputs, sizeof(float));
    memcpy(classification_loss_delta, l.delta, l.batch*l.outputs*sizeof(float));
    for (b = 0; b < l.batch; ++b) {
        for (j = 0; j < l.h; ++j) {
            for (i = 0; i < l.w; ++i) {
                for (n = 0; n < l.n; ++n) {

					index = entry_poly_index(l, b, n*l.w*l.h + j*l.w + i, 0);

					classification_loss_delta[index + 0*stride] = 0;
					classification_loss_delta[index + 1*stride] = 0;
					classification_loss_delta[index + 2*stride] = 0;
					classification_loss_delta[index + 3*stride] = 0;

					for (k = 0; k < l.poly_angles; ++k) {
						classification_loss_delta[index + (5 + l.classes + 3*k + 0)*stride] = 0;
						classification_loss_delta[index + (5 + l.classes + 3*k + 1)*stride] = 0;
						classification_loss_delta[index + (5 + l.classes + 3*k + 2)*stride] = 0;
					}
                }
            }
        }
    }
    float classification_loss = pow(mag_array(classification_loss_delta, l.outputs*l.batch), 2);
    free(classification_loss_delta);
    //
	float* poly_rlo_loss_delta = (float*)calloc(l.batch*l.outputs, sizeof(float));
	memcpy(poly_rlo_loss_delta, l.delta, l.batch*l.outputs*sizeof(float));
	for (b = 0; b < l.batch; ++b) {
        for (j = 0; j < l.h; ++j) {
            for (i = 0; i < l.w; ++i) {
                for (n = 0; n < l.n; ++n) {
                    
					index = entry_poly_index(l, b, n*l.w*l.h + j*l.w + i, 0);
					
					poly_rlo_loss_delta[index + 0*stride] = 0;
					poly_rlo_loss_delta[index + 1*stride] = 0;
					poly_rlo_loss_delta[index + 2*stride] = 0;
					poly_rlo_loss_delta[index + 3*stride] = 0;
					poly_rlo_loss_delta[index + 4*stride] = 0;

					for (k = 0; k < l.classes; ++k) {
						poly_rlo_loss_delta[index + (5 + k)*stride] = 0;
					}
					for (k = 0; k < l.poly_angles; ++k) {
						poly_rlo_loss_delta[index + (5 + l.classes + k*3 + 1)*stride] = 0;
						poly_rlo_loss_delta[index + (5 + l.classes + k*3 + 2)*stride] = 0;
					}
                }
            }
        }
    }
	float poly_rlo_loss = pow(mag_array(poly_rlo_loss_delta, l.outputs*l.batch), 2);
    free(poly_rlo_loss_delta);
	//
	float* poly_alpha_loss_delta = (float*)calloc(l.batch*l.outputs, sizeof(float));
	memcpy(poly_alpha_loss_delta, l.delta, l.batch*l.outputs*sizeof(float));
	for (b = 0; b < l.batch; ++b) {
        for (j = 0; j < l.h; ++j) {
            for (i = 0; i < l.w; ++i) {
                for (n = 0; n < l.n; ++n) {
                    index = entry_poly_index(l, b, n*l.w*l.h + j*l.w + i, 0);
					poly_alpha_loss_delta[index + 0*stride] = 0;
					poly_alpha_loss_delta[index + 1*stride] = 0;
					poly_alpha_loss_delta[index + 2*stride] = 0;
					poly_alpha_loss_delta[index + 3*stride] = 0;
					poly_alpha_loss_delta[index + 4*stride] = 0;
					for (k = 0; k < l.classes; ++k) {
						poly_alpha_loss_delta[index + (5 + k)*stride] = 0;
					}
					for (k = 0; k < l.poly_angles; ++k) {
						poly_alpha_loss_delta[index + (5 + l.classes + k*3 + 0)*stride] = 0;
						poly_alpha_loss_delta[index + (5 + l.classes + k*3 + 2)*stride] = 0;
					}
                }
            }
        }
    }
	float poly_alpha_loss = pow(mag_array(poly_alpha_loss_delta, l.outputs*l.batch), 2);
    free(poly_alpha_loss_delta);
	//
	float* poly_prob_loss_delta = (float*)calloc(l.batch*l.outputs, sizeof(float));
	memcpy(poly_prob_loss_delta, l.delta, l.batch*l.outputs*sizeof(float));
	for (b = 0; b < l.batch; ++b) {
        for (j = 0; j < l.h; ++j) {
            for (i = 0; i < l.w; ++i) {
                for (n = 0; n < l.n; ++n) {
                    index = entry_poly_index(l, b, n*l.w*l.h + j*l.w + i, 0);
					poly_prob_loss_delta[index + 0*stride] = 0;
					poly_prob_loss_delta[index + 1*stride] = 0;
					poly_prob_loss_delta[index + 2*stride] = 0;
					poly_prob_loss_delta[index + 3*stride] = 0;
					poly_prob_loss_delta[index + 4*stride] = 0;
					for (k = 0; k < l.classes; ++k) {
						poly_prob_loss_delta[index + (5 + k)*stride] = 0;
					}
					for (k = 0; k < l.poly_angles; ++k) {
						poly_prob_loss_delta[index + (5 + l.classes + k*3 + 0)*stride] = 0;
						poly_prob_loss_delta[index + (5 + l.classes + k*3 + 1)*stride] = 0;
					}
                }
            }
        }
    }
	float poly_prob_loss = pow(mag_array(poly_prob_loss_delta, l.outputs*l.batch), 2);
    free(poly_prob_loss_delta);
	//
	float loss = pow(mag_array(l.delta, l.outputs*l.batch), 2);
	//
    float iou_loss = loss - classification_loss - poly_rlo_loss - poly_alpha_loss - poly_prob_loss;
	//
    float avg_iou_loss = 0;
    // gIOU loss + MSE (objectness) loss
    if (l.iou_loss == MSE) {
        *(l.cost) = pow(mag_array(l.delta, l.outputs*l.batch), 2);
    } else {
        // Always compute classification loss both for iou + cls loss and for logging with mse loss
        // TODO: remove IOU loss fields before computing MSE on class probably split into two arrays
        if (l.iou_loss == GIOU) {
            avg_iou_loss = count > 0 ? l.iou_normalizer*(tot_giou_loss/count) : 0;
        } else {
            avg_iou_loss = count > 0 ? l.iou_normalizer*(tot_iou_loss/count) : 0;
        }
		*(l.cost) = avg_iou_loss + classification_loss + poly_rlo_loss + poly_alpha_loss + poly_prob_loss;
    }

    loss /= l.batch;
    classification_loss /= l.batch;
    iou_loss /= l.batch;
	poly_rlo_loss /= l.batch;
	poly_alpha_loss /= l.batch;
	poly_prob_loss /= l.batch;
	//
	static int file_flag = 0;
	if (file_flag == 0) {
		file_flag = 1;
		FILE *fp = fopen("lossinfo.txt", "wb");
		fclose(fp);
	}
	FILE *fp = fopen("lossinfo.txt","ab+");
	fprintf(fp, "loss=%0.4f,", loss);
	fprintf(fp, "classification_loss=%0.4f, ", classification_loss);
	fprintf(fp, "iou_loss=%0.4f, ", iou_loss);
	fprintf(fp, "avg_cat=%0.4f, avg_obj=%0.4f, avg_anyobj=%0.4f, recall=%0.4f,recall75=%0.4f, ", avg_cat/class_count, avg_obj/count, avg_anyobj/(l.w*l.h*l.n*l.batch), recall/count, recall75/count);
	fprintf(fp, "poly_rlo_loss=%0.4f, poly_alpha_loss=%0.4f, poly_prob_loss=%0.4f,", poly_rlo_loss, poly_alpha_loss, poly_prob_loss);
	fprintf(fp, "avg_poly_rlo=%0.4f, avg_poly_alpha=%0.4f, avg_poly_prob=%0.4f,", avg_poly_rlo/poly_count, avg_poly_alpha/poly_count, avg_poly_prob/poly_count);
	fprintf(fp, "\r\n");
	fclose(fp);

	printf("loss=%0.4f,", loss);
	printf("classification_loss=%0.4f, ", classification_loss);
	printf("iou_loss=%0.4f, ", iou_loss);
	printf("avg_cat=%0.4f, avg_obj=%0.4f, avg_anyobj=%0.4f, recall=%0.4f,recall75=%0.4f, ", avg_cat / class_count, avg_obj / count, avg_anyobj / (l.w*l.h*l.n*l.batch), recall / count, recall75 / count);
	printf("poly_rlo_loss=%0.4f, poly_alpha_loss=%0.4f, poly_prob_loss=%0.4f,", poly_rlo_loss, poly_alpha_loss, poly_prob_loss);
	printf("avg_poly_rlo=%0.4f, avg_poly_alpha=%0.4f, avg_poly_prob=%0.4f,", avg_poly_rlo / poly_count, avg_poly_alpha / poly_count, avg_poly_prob / poly_count);
	printf("\r\n");

	//fprintf(stderr, "v3 (%s loss, Normalizer: (iou: %.2f, cls: %.2f) Region %d Avg (IOU: %f, GIOU: %f), Class: %f, Obj: %f, No Obj: %f, .5R: %f, .75R: %f, count: %d, class_loss = %f, iou_loss = %f, total_loss = %f \n",
    //    (l.iou_loss == MSE ? "mse" : (l.iou_loss == GIOU ? "giou" : "iou")), l.iou_normalizer, l.cls_normalizer, state.index, tot_iou/count, tot_giou/count, avg_cat/class_count, avg_obj/count, avg_anyobj/(l.w*l.h*l.n*l.batch), recall/count, recall75/count, count,
    //    classification_loss, iou_loss, loss);

}

void backward_poly_layer(const layer l, network_state state)
{
   axpy_cpu(l.batch*l.inputs, 1, l.delta, 1, state.delta, 1);
}

// Converts output of the network to detection boxes
// w,h: image width,height
// netw,neth: network width,height
// relative: 1 (all callers seems to pass TRUE)
void correct_poly_boxes(detection *dets, int n, int w, int h, int netw, int neth, int relative, int letter, int poly_angles)
{
    int i,k;
    // network height (or width)
    int new_w = 0;
    // network height (or width)
    int new_h = 0;
    // Compute scale given image w,h vs network w,h
    // I think this "rotates" the image to match network to input image w/h ratio
    // new_h and new_w are really just network width and height
    if (letter) {
        if (((float)netw/w) < ((float)neth/h)) {
            new_w = netw;
            new_h = (h*netw)/w;
        } else {
            new_h = neth;
            new_w = (w*neth)/h;
        }
    }
    else {
        new_w = netw;
        new_h = neth;
    }
    // difference between network width and "rotated" width
    float deltaw = netw - new_w;
    // difference between network height and "rotated" height
    float deltah = neth - new_h;
    // ratio between rotated network width and network width
    float ratiow = (float)new_w/netw;
    // ratio between rotated network width and network width
    float ratioh = (float)new_h/neth;
    for (i = 0; i < n; ++i) {

        box b = dets[i].bbox;
        // x = (x - (deltaw/2)/netw)/ratiow;
        //   x - [(1/2 the difference of the network width and rotated width)/(network width)]
        b.x = (b.x - deltaw/2./netw)/ratiow;
        b.y = (b.y - deltah/2./neth)/ratioh;
        // scale to match rotation of incoming image
        b.w *= 1/ratiow;
        b.h *= 1/ratioh;

		for (k = 0; k < poly_angles; ++k) {
			dets[i].vers[k].x = (dets[i].vers[k].x - deltaw/2./netw)/ratiow;
			dets[i].vers[k].y = (dets[i].vers[k].y - deltah/2./neth)/ratioh;
		}

        // relative seems to always be == 1, I don't think we hit this condition, ever.
        if (!relative) {

            b.x *= w;
            b.w *= w;
            b.y *= h;
            b.h *= h;
			for (k = 0; k < poly_angles; ++k) {
				dets[i].vers[k].x *= w;
				dets[i].vers[k].y *= h;
			}
        }
        dets[i].bbox = b;
    }
}

int poly_num_detections(layer l, float thresh)
{
    int i, n;
    int count = 0;
    for(n = 0; n < l.n; ++n){
        for (i = 0; i < l.w*l.h; ++i) {
            int obj_index = entry_poly_index(l, 0, n*l.w*l.h + i, 4);
            if (l.output[obj_index] > thresh){
                ++count;
            }
        }
    }
    return count;
}

int poly_num_detections_batch(layer l, float thresh, int batch)
{
    int i, n;
    int count = 0;
    for (i = 0; i < l.w*l.h; ++i){
        for(n = 0; n < l.n; ++n){
            int obj_index = entry_poly_index(l, batch, n*l.w*l.h + i, 4);
            if (l.output[obj_index] > thresh){
                ++count;
            }
        }
    }
    return count;
}

int get_poly_detections(layer l, int w, int h, int netw, int neth, float thresh, int *map, int relative, detection *dets, int letter)
{
    //printf("\n l.batch = %d, l.w = %d, l.h = %d, l.n = %d \n", l.batch, l.w, l.h, l.n);
    int i,j,n;
    float *predictions = l.output;
    // This snippet below is not necessary
    // Need to comment it in order to batch processing >= 2 images
    //if (l.batch == 2) avg_flipped_yolo(l);
    int count = 0;
    for (i = 0; i < l.w*l.h; ++i){
        int row = i / l.w;
        int col = i % l.w;
        for(n = 0; n < l.n; ++n){
            int obj_index  = entry_poly_index(l, 0, n*l.w*l.h + i, 4);
            float objectness = predictions[obj_index];
            //if(objectness <= thresh) continue;    // incorrect behavior for Nan values
            if (objectness > thresh) {
                //printf("\n objectness = %f, thresh = %f, i = %d, n = %d \n", objectness, thresh, i, n);
                int box_index = entry_poly_index(l, 0, n*l.w*l.h + i, 0);
				poly_box bbox = get_poly_box(predictions, l.biases, l.mask[n], box_index, col, row, l.w, l.h, netw, neth, l.w*l.h, l.classes, l.poly_angles, l.poly_angle_step, 1);
				dets[count].bbox.x = bbox.x;
				dets[count].bbox.y = bbox.y;
				dets[count].bbox.w = bbox.w;
				dets[count].bbox.h = bbox.h;
				for (j = 0; j < l.poly_angles; ++j) {
					dets[count].vers[j].x = bbox.prs[j].x;
					dets[count].vers[j].y = bbox.prs[j].y;
					dets[count].vers[j].prob = bbox.prs[j].prob;
				}
                dets[count].objectness = objectness;
                dets[count].classes = l.classes;
                for (j = 0; j < l.classes; ++j) {
                    int class_index = entry_poly_index(l, 0, n*l.w*l.h + i, 4 + 1 + j);
                    float prob = objectness*predictions[class_index];
                    dets[count].prob[j] = (prob > thresh) ? prob : 0;
                }
                ++count;
            }
        }
    }
    correct_poly_boxes(dets, count, w, h, netw, neth, relative, letter, l.poly_angles);
    return count;
}

int get_poly_detections_batch(layer l, int w, int h, int netw, int neth, float thresh, int *map, int relative, detection *dets, int letter, int batch)
{
    int i,j,n;
    float *predictions = l.output;
    //if (l.batch == 2) avg_flipped_yolo(l);
    int count = 0;
    for (i = 0; i < l.w*l.h; ++i){
        int row = i / l.w;
        int col = i % l.w;
        for(n = 0; n < l.n; ++n){
            int obj_index  = entry_poly_index(l, batch, n*l.w*l.h + i, 4);
            float objectness = predictions[obj_index];
            //if(objectness <= thresh) continue;    // incorrect behavior for Nan values
            if (objectness > thresh) {
                //printf("\n objectness = %f, thresh = %f, i = %d, n = %d \n", objectness, thresh, i, n);
                int box_index = entry_poly_index(l, batch, n*l.w*l.h + i, 0);
				poly_box bbox = get_poly_box(predictions, l.biases, l.mask[n], box_index, col, row, l.w, l.h, netw, neth, l.w*l.h, l.classes, l.poly_angles, l.poly_angle_step, 1);
				dets[count].bbox.x = bbox.x;
				dets[count].bbox.y = bbox.y;
				dets[count].bbox.w = bbox.w;
				dets[count].bbox.h = bbox.h;
				for (j = 0; j < l.poly_angles; ++j) {
					dets[count].vers[j].x = bbox.prs[j].x;
					dets[count].vers[j].y = bbox.prs[j].y;
					dets[count].vers[j].prob = bbox.prs[j].prob;
				}
                dets[count].objectness = objectness;
                dets[count].classes = l.classes;
                for (j = 0; j < l.classes; ++j) {
                    int class_index = entry_poly_index(l, batch, n*l.w*l.h + i, 4 + 1 + j);
                    float prob = objectness*predictions[class_index];
                    dets[count].prob[j] = (prob > thresh) ? prob : 0;
                }
                ++count;
            }
        }
    }
    correct_poly_boxes(dets, count, w, h, netw, neth, relative, letter, l.poly_angles);
    return count;
}

#ifdef GPU
void forward_poly_layer_gpu(const layer l, network_state state)
{
	int b, n, k, index;
    //copy_ongpu(l.batch*l.inputs, state.input, 1, l.output_gpu, 1);
    simple_copy_ongpu(l.batch*l.inputs, state.input, l.output_gpu);
    for (b = 0; b < l.batch; ++b) {
        for (n = 0; n < l.n; ++n) {
            index = entry_poly_index(l, b, n*l.w*l.h, 0);
            // y = 1./(1. + exp(-x))
            // x = ln(y/(1-y))  // ln - natural logarithm (base = e)
            // if(y->1) x -> inf
            // if(y->0) x -> -inf
            activate_array_ongpu(l.output_gpu + index, 2*l.w*l.h, LOGISTIC);    // x,y
            if (l.scale_x_y != 1) scal_add_ongpu(2*l.w*l.h, l.scale_x_y, -0.5*(l.scale_x_y - 1), l.output_gpu + index, 1);      // scale x,y
            index = entry_poly_index(l, b, n*l.w*l.h, 4);
            activate_array_ongpu(l.output_gpu + index, (1+l.classes)*l.w*l.h, LOGISTIC); // classes and objectness
			for (k = 0; k < l.poly_angles; ++k) {
				index = entry_poly_index(l, b, n*l.w*l.h, 4 + 1 + l.classes + k*3 + 1);
				activate_array_ongpu(l.output + index, 2*l.w*l.h, LOGISTIC);
			}
        }
    }
    if (!state.train || l.onlyforward) {
        //cuda_pull_array(l.output_gpu, l.output, l.batch*l.outputs);
        if (l.mean_alpha && l.output_avg_gpu) mean_array_gpu(l.output_gpu, l.batch*l.outputs, l.mean_alpha, l.output_avg_gpu);
        cuda_pull_array_async(l.output_gpu, l.output, l.batch*l.outputs);
        CHECK_CUDA(cudaPeekAtLastError());
        return;
    }

    float *in_cpu = (float*)xcalloc(l.batch*l.inputs, sizeof(float));
    cuda_pull_array(l.output_gpu, l.output, l.batch*l.outputs);
    memcpy(in_cpu, l.output, l.batch*l.outputs*sizeof(float));
    float *truth_cpu = 0;
    if (state.truth) {
        int num_truth = l.batch*l.truths;
        truth_cpu = (float*)xcalloc(num_truth, sizeof(float));
        cuda_pull_array(state.truth, truth_cpu, num_truth);
    }
    network_state cpu_state = state;
    cpu_state.net = state.net;
    cpu_state.index = state.index;
    cpu_state.train = state.train;
    cpu_state.truth = truth_cpu;
    cpu_state.input = in_cpu;
    forward_poly_layer(l, cpu_state);
    cuda_push_array(l.delta_gpu, l.delta, l.batch*l.outputs);
    free(in_cpu);
    if (cpu_state.truth) free(cpu_state.truth);
}

void backward_poly_layer_gpu(const layer l, network_state state)
{
    axpy_ongpu(l.batch*l.inputs, state.net.loss_scale, l.delta_gpu, 1, state.delta, 1);
}
#endif
