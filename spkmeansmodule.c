#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdio.h>
#include <stdlib.h>
#include <spkmeans.h>
#include <string.h>

/*
The function finds the index of the closest centroid to the array x.
The distance is measured using the euclidean distance.
The function assumes the dimension of the centroids and of x is d.
The function uses the function euclid_dist_sq.
*/
static int find_closest(double** centroids, double* x, int k, int ddim){
    double minimal_distance, curr_distance;
    int minimal_index, i;
    
    curr_distance = minimal_distance = euclid_dist(centroids[0], x, d);
    minimal_index = 0;
    for (i = 1; i < k; i++){
         curr_distance = euclid_dist(centroids[i], x, d);
         if (minimal_distance > curr_distance){
             minimal_distance = curr_distance;
             minimal_index = i;
         }
    }
    return minimal_index;
}

/*
The function calculates the k cluster centroids produced by the K-means algorithm on the observations.
The function receives the k points to serve as the centroids in the first iteration.
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
static int calculate_kmeans(double** obs, double** centroids, int n, int d, int k,
 int max_iter, double epsilon){
    double** new_centroids;
    int* cluster_counts;
    int i, j, curr_index, converged;
    
    new_centroids = calloc(k, sizeof(double*));
    if (new_centroids == NULL){ return 1; }
    cluster_counts = calloc(k, sizeof(int));
    if (cluster_counts == NULL){
        free(new_centroids);
        return 1;
    }
    for (i = 0; i < k; i++){
        new_centroids[i] = calloc(d, sizeof(double));
        if (new_centroids[i] == NULL){
            free_matrix(new_centroids, i);
            free(cluster_counts);
            return 1;
        }
    }
    converged = 0;
    while (converged == 0 && max_iter > 0)
    {
        for (i = 0; i < n; i++){
            curr_index = find_closest(centroids, obs[i], k, d);
            cluster_counts[curr_index] += 1;
            for (j = 0; j < d; j++){
                new_centroids[curr_index][j] += obs[i][j];
            }
        }
        converged = 1;
        for (i = 0; i < k; i++){
            if (cluster_counts[i] == 0){
                free_matrix(new_centroids, k);
                free(cluster_counts);
                return 1;
            }
            for (j = 0; j < d; j++){
                new_centroids[i][j] = new_centroids[i][j] / cluster_counts[i];
            }
            if (euclid_dist(new_centroids[i], centroids[i], d) >= epsilon){
                converged = 0;
            }
        }
        max_iter--;
        for (i = 0; i < k; i++){
            cluster_counts[i] = 0;
            for (j = 0; j < d; j++){
                centroids[i][j] = new_centroids[i][j];
                new_centroids[i][j] = 0;
            }
        }
    }
    free_matrix(new_centroids, k);
    free(cluster_counts);
    return 0;
}

/*
The function receives a Python list of float vectors.
The function returns an equivalent double C matrix.
The assumed number of elements is n, and the assumed dimension is d.
*/
static double** read_from_python(int n, int d, PyObject *python_list){
    int i, j;
    double **matrix;
    PyObject *temp_list, *element;
    matrix = calloc(n, sizeof(double*));
    if (matrix == NULL){
        free(matrix);
        return NULL;
    }
    for (i = 0; i < n; i++){
        matrix[i] = calloc(d, sizeof(double));
        if (matrix[i] == NULL){
            free_matrix(matrix, i);
            return NULL;
        }
        temp_list = PyList_GetItem(python_list, i);
        for (j = 0; j < d; j++){
            element = PyList_GetItem(temp_list, j);
            matrix[i][j] = PyFloat_AsDouble(element);
        }
    }
    return matrix;
}

/*
The function receives a C array of double vectors.
The function returns an equivalent float Python matrix.
The assumed number of elements is n, and the assumed dimension is d.
*/
static PyObject* write_to_python(double** mat, int n, int d){
    int i, j;
    PyObject* outer_list;
    PyObject* inner_list;
    PyObject* element;
    outer_list = PyList_New(N);
    for (i = 0; i < n; i++){
        inner_list = PyList_New(dim);
        for (j = 0; j < d; j++){
            element = PyFloat_FromDouble(mat[i][j]);
            PyList_SET_ITEM(inner_list, j, element);
        }
        PyList_SET_ITEM(outer_list, i, inner_list);
    }
    return outer_list;
}

static PyObject* apply_mat_ops(PyObject *args){
    double** mat, **result;
    PyObject* python_mat;
    char* goal;
    int n, d;

    if (!PyArg_ParseTuple(args, "siiO", &goal, &n, &d, &python_mat)){ return NULL; }
    
    mat = read_from_python(n, d, python_mat);
    if (mat == NULL){ return NULL; }
    
    result = calculate_mat(mat, goal, n, d);
    free_matrix(mat, n);
    
    if (result == NULL){ return NULL; }
    return write_to_python(mat, n + (strcmp(goal, "jacobi") == 0), d);
}

static PyObject* apply_kmeans_prep(PyObject *args){}

static PyObject* apply_kmeans(PyObject *args){
    int n, k, max_iter, d;
    double eps;
    char* goal;
    PyObject *centroid_list, *observation_list, *result;
    double** centroids, **observations

    if (!PyArg_ParseTuple(args, "siiiidOO", &goal, &n, &k, &max_iter, &d, &eps, 
        &centroid_list, &observation_list)){
            return NULL;
        }
    centroids = read_from_python(k, d, centroid_list);
    observations = read_from_python(n, d, observation_list);
    if ((centroids == NULL) || (observations == NULL) || 
    (calculate_kmeans(observations, centroids, n, d, k, max_iter, eps) == 1)) {
        result = NULL;
    }
    else { result = write_to_python(centroids, k, d);}
    
    free_matrix(observations, n);
    free_matrix(centroids, k);
    return result;
}

static PyObject* apply(PyObject *self, PyObject *args) {
    char* goal;
    
    if (!PyArg_ParseTuple(args, "s", &goal)){ return NULL; }
    
    if ((strcmp(goal, "kmeans_prep") != 0) && (strcmp(goal, "kmeans") != 0)){
        return apply_mat_ops(args);
    }
    
    if (strcmp(goal, "kmeans_prep") == 0){
        return apply_kmeans_prep(args);
    }

    else{ return apply_kmeans(args); }
}

/*
Python module setup
*/
static PyMethodDef capiMethods[] = {
    {"apply", (PyCFunction) apply, METH_VARARGS, NULL},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT, 
    "spkmeansmodule",
    NULL, 
    -1,
    capiMethods
};

PyMODINIT_FUNC
PyInit_spkmeansmoudle(void)
{
    PyObject *m;
    m = PyModule_Create(&moduledef);
    if (!m){
        return NULL;
    }
    return m;
}