#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdio.h>
#include <stdlib.h>
#include <spkmeans.h>
#include <string.h>

/*
The function finds the index of the closest centroid to the array x.
The distance is measured using the euclidean distance.
The function assumes the dimension of the centroids and of x is dim.
The function uses the function euclid_dist_sq.
*/
static int find_closest(double** centroids, double* x, int K, int dim){
    double minimal_distance, curr_distance;
    int minimal_index, i;
    
    curr_distance = minimal_distance = euclid_dist(centroids[0], x, dim);
    minimal_index = 0;
    for (i = 1; i < K; i++){
         curr_distance = euclid_dist(centroids[i], x, dim);
         if (minimal_distance > curr_distance){
             minimal_distance = curr_distance;
             minimal_index = i;
         }
    }
    return minimal_index;
}

/*
The function calculates the K cluster centroids produced by the K-means algorithm on the observations.
The function receives the K points to serve as the centroids in the first iteration.
The function then iterates, performing the following:
- Adding each observation's elements to the sums of its closest cluster, and incrementing
  the updated size of the cluster. 
  Figuring out which is the closest cluster is done using the function find_closest.
- Calculating each cluster's new centroid as the average of the cluster's updated observations.
- Calculating each cluster's deviation between its old centroid and its new one. The deviation is calculated
using the function euclid_dist_sq.
The function stops when either max_iter iterations have happend, or when the deviation of any cluster
is less than epsilon squared (the distance itself is less than epsilon).
*/
static int calculate_kmeans(double** obs, double** centroids, int N, int dim, int K, int max_iter, double epsilon){
    double** new_centroids;
    int* cluster_counts;
    int i, j, curr_index, converged;
    
    new_centroids = calloc(K, sizeof(double*));
    cluster_counts = calloc(K, sizeof(int));
    for (i = 0; i < K; i++){
        new_centroids[i] = calloc(dim, sizeof(double));
    }
    converged = 0;
    while (converged == 0 && max_iter > 0)
    {
        for (i = 0; i < N; i++){
            curr_index = find_closest(centroids, obs[i], K, dim);
            cluster_counts[curr_index] += 1;
            for (j = 0; j < dim; j++){
                new_centroids[curr_index][j] += obs[i][j];
            }
        }
        converged = 1;
        for (i = 0; i < K; i++){
            if (cluster_counts[i] == 0){
                free_matrix(new_centroids, K);
                free(cluster_counts);
                return 1;
            }
            for (j = 0; j < dim; j++){
                new_centroids[i][j] = new_centroids[i][j] / cluster_counts[i];
            }
            if (euclid_dist(new_centroids[i], centroids[i], dim) >= (epsilon * epsilon)){
                converged = 0;
            }
        }
        max_iter--;
        for (i = 0; i < K; i++){
            cluster_counts[i] = 0;
            for (j = 0; j < dim; j++){
                centroids[i][j] = new_centroids[i][j];
                new_centroids[i][j] = 0;
            }
        }
    }
    free_matrix(new_centroids, K);
    free(cluster_counts);
    return 0;
}

/*
The function receives a Python list of float vectors.
The function returns an equivalent double C matrix.
The assumed number of elements is num_of_elements, and the assumed dimension is dim.
*/
static double** read_from_python(int num_of_elements, int dim, PyObject *python_list){
    int i, j;
    double **matrix;
    PyObject *temp_list, *element;
    matrix = calloc(num_of_elements, sizeof(double*));
    if (matrix == NULL){
        free(matrix);
        return NULL;
    }
    for (i = 0; i < num_of_elements; i++){
        matrix[i] = calloc(dim, sizeof(double));
        if (matrix[i] == NULL){
            free_matrix(matrix, i);
            return NULL;
        }
        temp_list = PyList_GetItem(python_list, i);
        for (j = 0; j < dim; j++){
            element = PyList_GetItem(temp_list, j);
            matrix[i][j] = PyFloat_AsDouble(element);
        }
    }
    return matrix;
}

/*
The function receives a C array of double vectors.
The function returns an equivalent float Python matrix.
The assumed number of elements is N, and the assumed dimension is dim.
*/
static PyObject* write_to_python(double** mat, int N, int dim){
    int i, j;
    PyObject* outer_list;
    PyObject* inner_list;
    PyObject* element;
    outer_list = PyList_New(N);
    for (i = 0; i < N; i++){
        inner_list = PyList_New(dim);
        for (j = 0; j < dim; j++){
            element = PyFloat_FromDouble(mat[i][j]);
            PyList_SET_ITEM(inner_list, j, element);
        }
        PyList_SET_ITEM(outer_list, i, inner_list);
    }
    return outer_list;
}

static PyObject* apply_mat_ops(PyObject *args){
    int N, d;
    char* goal;
    PyObject *observation_list;
    double** observations;
    
    if (!PyArg_ParseTuple(args, "s", &goal)){
        return NULL;
    }
    
    if (!PyArg_ParseTuple(args, "siiO", &goal, &N, &dim, &observation_list)){
            return NULL;
        }
    observations = read_from_python(N, dim, observation_list);
    if (observations == NULL){
        return NULL;
    }
        
    if (strcmp(goal, "jacobi") == 0){
        jacobi = jacobi_eval_evec(observations, N);
        result = write_to_python(jacobi, N+1, N);
        free_matrix(jacobi, N+1);
        return result;
    }
        
    wam = weighted_adj_mat(observations, N, dim);
    if (wam == NULL){ return NULL; }
    if (strcmp(goal, "wam") == 0){
        result = write_to_python(wam, N, N);
        free_matrix(wam, N);
        return result;
        }
    else{
        ddg = diag_deg_mat(wam, N);
        if ((ddg == NULL) || (strcmp(goal, "ddg") == 0)){
            free_matrix(wam, N);
            if (ddg == NULL){
                return NULL;
            }
            result = write_to_python(ddg, N, N);
            free_matrix(ddg, N);
            return result;
        }
        else{
            lnorm = norm_graph_lap(wam, ddg, N);
            free_matrix(wam, N);
            free_matrix(ddg, N);
            if (lnorm == NULL){ return NULL; }
            result = write_to_python(lnorm, N, N);
            free_matrix(lnorm, N);
            return result;
        }
    }
}

static PyObject* apply_kmeans_prep(PyObject *args){}

static PyObject* apply_kmeans(PyObject *args){
    int N, K, max_iter, dim;
    double eps;
    char* goal;
    PyObject *centroid_list, *observation_list, *result;
    double** centroids, **observations

    if (!PyArg_ParseTuple(args, "s", &goal)){
        return NULL;
    }

    if (!PyArg_ParseTuple(args, "iiiidOO", &N, &K, &max_iter, &dim, &eps, 
        &centroid_list, &observation_list)){
            return NULL;
        }
    centroids = read_from_python(K, dim, centroid_list);
    observations = read_from_python(N, dim, observation_list);
    if (calculate_kmeans(observations, centroids, N, dim, K, max_iter, eps) == 1)
    {
        free_matrix(observations, N);
        free_matrix(centroids, K);
        return NULL;
    }
    else {
        result = write_to_python(centroids, K, dim);
        free_matrix(observations, N);
        free_matrix(centroids, K);
        return result;
    }
}


/*
The function receives the following arguments needed for the kmeans algorithm:
N - num of observations, K - num of centroids, max_iter, dim - dimension of observations,
eps - convergence bound, centroid_list - first centroids (randomly calculatd in Python),
observation_list - observations.
The function converts the required data to be usable in C, applies the kmeans algorithm, 
converts the results back to be usable in python and returns them.
*/
static PyObject* apply(PyObject *self, PyObject *args) {
    char* goal;
    
    if (!PyArg_ParseTuple(args, "s", &goal)){
        return NULL;
    }
    
    if ((strcmp(goal, "kmeans_prep") != 0) && (strcmp(goal, "kmeans") != 0)){
        return apply_mat_ops(args);
    }
    
    if (strcmp(goal, "kmeans_prep") == 0){
        return apply_kmeans_prep(args);
    }

    else{
        return apply_kmeans(args);
    }
}

/*
Python module setup
*/
static PyMethodDef capiMethods[] = {
    {"fit", (PyCFunction) fit, METH_VARARGS, NULL},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT, 
    "mykmeanssp",
    NULL, 
    -1,
    capiMethods
};

PyMODINIT_FUNC
PyInit_mykmeanssp(void)
{
    PyObject *m;
    m = PyModule_Create(&moduledef);
    if (!m){
        return NULL;
    }
    return m;
}