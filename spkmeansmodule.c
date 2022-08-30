#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdio.h>
#include <stdlib.h>
#include "spkmeans.h"
#include <string.h>
#include <math.h>

/*
The function finds the index of the closest centroid to the array x.
The distance is measured using the euclidean distance.
The function assumes the dimension of the centroids and of x is d.
The function uses the function euclid_dist.
*/
static int find_closest(double** centroids, double* x, int k, int d){
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
    
    new_centroids = (double**) calloc(k, sizeof(double*));
    if (new_centroids == NULL){ return 1; }
    cluster_counts = (int*) calloc(k, sizeof(int));
    if (cluster_counts == NULL){
        free(new_centroids);
        return 1;
    }
    for (i = 0; i < k; i++){
        new_centroids[i] = (double*) calloc(d, sizeof(double));
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
    matrix = (double**) calloc(n, sizeof(double*));
    if (matrix == NULL){
        free(matrix);
        return NULL;
    }
    for (i = 0; i < n; i++){
        matrix[i] = (double*) calloc(d, sizeof(double));
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
    outer_list = PyList_New(n);
    for (i = 0; i < n; i++){
        inner_list = PyList_New(d);
        for (j = 0; j < d; j++){
            element = PyFloat_FromDouble(mat[i][j]);
            PyList_SET_ITEM(inner_list, j, element);
        }
        PyList_SET_ITEM(outer_list, i, inner_list);
    }
    return outer_list;
}

/*
The function receives a goal, and a Python matrix of size n*d.
The function applies the desired goal to the matrix and returns it (in Python form).
*/
static PyObject* apply_mat_ops(PyObject *self, PyObject *args){
    double** mat, **result;
    PyObject *python_mat, *python_result;
    char* goal;
    int n, d, jacobi_flag;
    if (!PyArg_ParseTuple(args, "siiO", &goal, &n, &d, &python_mat)){ return NULL; }
    mat = read_from_python(n, d, python_mat);
    if (mat == NULL){ return NULL; }
    result = calculate_mat(mat, goal, n, d);
    free_matrix(mat, n);
    
    if (result == NULL){ return NULL; }
    jacobi_flag = (strcmp(goal, "jacobi") == 0); /* A boolean flag for jacobi specific use */
    python_result = write_to_python(result, n + jacobi_flag, n);
    free_matrix(result, n + jacobi_flag);
    return python_result;
}

/*
The function receives a double matrix of size n*d.
The function returns the transpose of said matrix.
*/
static double** transpose(double** mat, int n, int d){
    double** result;
    int i, j;

    result = (double**) calloc(d, sizeof(double*));
    if (result == NULL){ return NULL; }
    for (i = 0; i < d; i++){
        result[i] = (double*) calloc(n, sizeof(double));
        if (result[i] == NULL){
            free_matrix(result, i);
            return NULL;
        }
        for (j = 0; j < n; j++){
            result[i][j] = mat[j][i];
        }
    }
    return result;
}

/*
The function compares between two double arrays.
The first array is lesser than the second iff its first element is greater or equal than 
that of the second array.
*/
static int comparator(const void *x, const void *y)
{
    double* dx = (*(double**)x); /* Cast x to a pointer of a double array, and return the array */
    double* dy = (*(double**)y); /* Cast y to a pointer of a double array, and return the array */
    int left = dx[0] > dy[0];
    int right = dx[0] < dy[0];
    return right - left;
}

/*
The function receives the tranpose of the Jacobi matrix.
The first column is comprised of the e-vals, and the rest of each row is the corresponding
e-vector. 
The function sorts (using qsort) the matrix by the first element of each row (by e-vals).
The sorting is done decreasingly (larger e-vals first).
*/
static void sort_by_eval(double** jacobi_t, int n){
    qsort(jacobi_t, n, sizeof(jacobi_t[0]), comparator);
}

/*
The function receives the tranpose of the Jacobi matrix.
The first column is comprised of the e-vals, and the rest of each row is the corresponding
e-vector. 
The function calculates the largest eigen gap between the sorted e-vals, and returns the
appropriate amount of centroids to take.
*/
static int eigen_gap(double** jacobi_t, int n){
    int i, imax;
    double delta;

    delta = 0.0;
    imax = 0;
    for (i = 0; i < (n/2); i++){
        if (fabs(jacobi_t[i][0]-jacobi_t[i+1][0]) > delta){
            delta = fabs(jacobi_t[i][0]-jacobi_t[i+1][0]);
            imax = i;
        }
    }
    return imax+1; /* Amount of e-vals desired, not the index of the last */
}

/*
The function receives a double matrix of size n*d.
The function normalizes each row (by the square root of the sum of its squares).
*/
static void normalize(double** mat, int n, int d){
    int i, j;
    double s;

    for (i = 0; i < n; i++){
        s = 0.0;
        for (j = 0; j < d; j++){
            s += mat[i][j] * mat[i][j];
        }
        s = sqrt(s);
        for (j = 0; j < d; j++){
            if (s != 0){ mat[i][j] /= s; }
            else{ mat[i][j] = 0; }
        }
    }
}

/*
The function receives a Python jacobi matrix with n e-vectors, and a k value.
The function decreasingly sorts the columns by their first elements (e-vals).
If the k value received is 0, the function also applies the eigen-gap heuristic.
Then, the function returns the k first columns of the sorted matrix after normalization.
*/
static PyObject* apply_kmeans_prep(PyObject *self, PyObject *args){
    int n, k;
    PyObject *python_jacobi, *result;
    double** jacobi, **jacobi_t;

    if (!PyArg_ParseTuple(args, "iiO", &n, &k, &python_jacobi)){ return NULL; }
    jacobi = read_from_python(n+1, n, python_jacobi);
    if (jacobi == NULL) { return NULL; }

    jacobi_t = transpose(jacobi, n+1, n);
    if (jacobi_t == NULL) { result = NULL; }
    else{
        sort_by_eval(jacobi_t, n);
        if (k == 0){ k = eigen_gap(jacobi_t, n); }
        free_matrix(jacobi, n+1);
        jacobi = transpose(jacobi_t, n, n+1); /* Jacobi is now the extended U matrix */
        if (jacobi == NULL){ result = NULL; }
        else{
            normalize(jacobi + 1, n, k); /* Normalize to receive T */
            result = write_to_python(jacobi + 1, n, k);
        }
    }
    free_matrix(jacobi, n+1);
    free_matrix(jacobi_t, n);
    return result;
}

/*
The function receives the following arguments needed for the kmeans algorithm:
n - num of observations, k - num of centroids, max_iter, d - dimension of observations,
eps - convergence bound, centroid_list - first centroids (randomly calculatd in Python),
observation_list - observations.
The function converts the required data to be usable in C, applies the kmeans algorithm, 
converts the results back to be usable in python and returns them.
*/
static PyObject* apply_kmeans(PyObject *self, PyObject *args){
    int n, k, max_iter, d;
    double eps;
    PyObject *centroid_list, *observation_list, *result;
    double** centroids, **observations;

    if (!PyArg_ParseTuple(args, "iiiidOO", &n, &d, &k, &max_iter, &eps, 
        &centroid_list, &observation_list)){
            return NULL;
        }
    centroids = read_from_python(k, d, centroid_list);
    observations = read_from_python(n, d, observation_list);
    if ((centroids == NULL) || (observations == NULL) || 
    (calculate_kmeans(observations, centroids, n, d, k, max_iter, eps) == 1)) {
        result = NULL;
    }
    else { result = write_to_python(centroids, k, d); }
    
    free_matrix(observations, n);
    free_matrix(centroids, k);
    return result;
}

/*
Python module setup
*/
static PyMethodDef capiMethods[] = {
    {"apply_mat_ops", (PyCFunction) apply_mat_ops, METH_VARARGS, NULL},
    {"apply_kmeans_prep", (PyCFunction) apply_kmeans_prep, METH_VARARGS, NULL},
    {"apply_kmeans", (PyCFunction) apply_kmeans, METH_VARARGS, NULL},
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
PyInit_spkmeansmodule(void)
{
    PyObject *m;
    m = PyModule_Create(&moduledef);
    if (!m){
        return NULL;
    }
    return m;
}