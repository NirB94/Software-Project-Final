#define sign(x) (x < 0 ? -1 : 1)
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <math.h>
#include <string.h>
#include "spkmeans.h"


void free_matrix(double** mat, int n) {
    int i;
    
    if (mat != NULL) {
        for (i = 0; i < n; i++) {
            free(mat[i]);
        }
        free(mat);
    }
}

/*
The function check if the input is of the right length.
Then, the function checks whether goal is valid.
*/
int first_input_validation(int length_of_input, char *input[])
{
    if (length_of_input != 3){
        printf("Invalid Input!\n");
        return 1;
    }
    if (strcmp(input[1], "wam") != 0 && strcmp(input[1], "ddg") != 0 && strcmp(input[1], "lnorm") != 0 &&
    strcmp(input[1], "jacobi") != 0){
        printf("Invalid Input!\n");
        return 1;
    }
    return 0;
}

/*
The function retrieves the dimensions of the input file.
The function inputs said dimensions (# of rows and columns) into the dims array.
*/
int find_dimensions(char const *filename, int *dims){
    FILE *f = NULL;
    char c;

    f = fopen(filename, "r");
    if (f == NULL) {
        fclose(f);
        printf("An Error Has Occurred\n");
        return 1;
    }
    dims[0] = 0;
    dims[1] = 1;
    while ((c = fgetc(f)) != EOF) 
    {
        if (c == '\n')
        {
            dims[0]++;
        }
        else if (c == ',')
        {
            if (dims[0] == 0)
            {
                dims[1]++;
            }
        }
    }
    fclose(f);
    return 0;
}

/*
The function reads the input file.
The function returns a double matrix of size n * d whose elements are that of the input file.
*/
double** read_file(char const *filename, int n, int d) {
    FILE *f = NULL;
    char c;
    int i, j;
    double** obs;

    f = fopen(filename, "r");
    if (f == NULL) {
        fclose(f);
        printf("An Error Has Occurred\n");
        return NULL;
    }
    obs = (double**) calloc(n, sizeof(double*));
    if (obs == NULL) {
        printf("An Error Has Occurred\n");
        return NULL;
    }
    for (i = 0; i < n; i++)
    {
        obs[i] = (double*) calloc(d, sizeof(double));
        if (obs[i] == NULL) {
            free_matrix(obs, i);
            printf("An Error Has Occurred\n");
            return NULL;
        }
        for (j = 0; j < d; j++)
        {
            fscanf(f, "%lf%c", &obs[i][j], &c);
        }
    }
    fclose(f);
    return obs;
}

/*
The function returns the euclidean distance between the two double vectors.
The function assumes the dimension of the vectors is d.
*/
double euclid_dist(double* x, double* y, int d){
    double dist;
    int j;

    dist = 0;
    for (j = 0; j < d; j++){
        dist += (x[j]-y[j]) * (x[j]-y[j]);
    }
    return sqrt(dist);
}

/*
The function receives an array of n observations with d elements each (of type double).
The function calculates the weighted adjacency matrix of the observations.
*/
double** weighted_adj_mat(double** obs, int n, int d){
    double** wam;
    int i, j;

    wam = (double**) calloc(n, sizeof(double*));
    if (wam == NULL) {
        return NULL;
    }
    for (i = 0; i < n; i++){
        wam[i] = (double*) calloc(n, sizeof(double));
        if (wam[i] == NULL) { 
            free_matrix(wam, i);
            return NULL;
        }
        wam[i][i] = 0;
    }
    for (i = 0; i < n; i++) {    
        for (j = i + 1; j < n; j++){
            wam[i][j] = exp(- euclid_dist(obs[i], obs[j], d) / 2.0);
            wam[j][i] = wam[i][j];
        }
    }
    return wam;
}

/*
The function receives a weighted adjacency matrix of size n*n.
The function calculates the apropriate diagonal degree matrix.
*/
double** diag_deg_mat(double** wam, int n){
    double** ddg;
    int i, j;

    ddg = (double**) calloc(n, sizeof(double*));
    if (ddg == NULL) {
        return NULL;
    }
    for (i = 0; i < n; i++){
        ddg[i] = (double*) calloc(n, sizeof(double));
        if (ddg[i] == NULL) {
            free_matrix(ddg, i);
            return NULL;
        }
        for (j = 0; j < n; j++){
            ddg[i][i] += wam[i][j];
        }
    }
    return ddg;
}

/*
The function receives a weighted adjacency matrix and its apropriate diagonal degree matrix.
Both matrices are of size n*n.
The function calculates the apropriate normalized graph laplacian, according to the following:
LNORM = I - DDG^(-0.5) * WAM * DDG^(-0.5)
*/
double** norm_graph_lap(double** wam, double** ddg, int n){
    double** lnorm;
    int i, j;

    for (i = 0; i < n; i++){
        ddg[i][i] = 1 / sqrt(ddg[i][i]);
    }
    
    lnorm = (double**) calloc(n, sizeof(double*));
    if (lnorm == NULL) {
        return NULL;
    }
    for (i = 0; i < n; i++){
        lnorm[i] = (double*) calloc(n, sizeof(double));
        if (lnorm[i] == NULL) { 
            free_matrix(lnorm, i);
            return NULL;
        }
        lnorm[i][i] = 1;
    }
    for (i = 0; i < n; i++){
        for (j = i + 1; j < n; j++){
            lnorm[i][j] = - (wam[i][j] * ddg[i][i] * ddg[j][j]);
            lnorm[j][i] = lnorm[i][j];
        }
    }
    for (i = 0; i < n; i++){ /* Restoring the received ddg (even though it isn't used later)*/
        ddg[i][i] = 1 / (ddg[i][i] * ddg[i][i]);
    }
    return lnorm;
}

/*
The function receives a symmetric double matrix of size n*n.
The function finds the pair of indices of the maximal off-diagonal element (in absolute terms).
*/
int* max_abs_off_diag(double** mat, int n){
    int i, j;
    int *max_indices;
    
    max_indices = (int*) calloc(2, sizeof(int));
    if (max_indices == NULL) {
        return NULL;
    }
    max_indices[0] = 0;
    max_indices[1] = 1;
    for (i = 0; i < n; i++){
        for (j = 0; j < n; j++){
            if (i != j){
                if (fabs(mat[max_indices[0]][max_indices[1]]) < fabs(mat[i][j])){
                    max_indices[0] = i;
                    max_indices[1] = j;
                }
            }
        }
    }
    return max_indices;
}

/*
The function receives a symmetric double matrix of size n*n.
The function calculates the sum of its squared off-diagonal elements.
*/
double sum_off_diag_sq(double** mat, int n){
    int i, j;
    double s = 0;

    for (i = 0; i < n; i++){
        for (j = 0; j < n; j++){
            if (i != j){
                s += pow(mat[i][j], 2);
            }
        }
    }
    return s;
}

/*
The function receives the n*n sized e-vector matrix V, the indices i & j, and the values c & s.
The function multiples V[1:, :] = V[1:, :] * P.
The first row of V is left unchanged.
The matrix P is given as described in the instructions.
*/
int update_e_vector_mat(double** V, double c, double s, int n, int i, int j){
    double* v1, *v2;
    int k;
    
    v1 = (double*) calloc(n, sizeof(double));
    if (v1 == NULL) {
        return 1;
    }
    v2 = (double*) calloc(n, sizeof(double));
    if (v2 == NULL) {
        return 1;
    }    
    for (k = 1; k < n+1; k++){
            v1[k-1] = (c * V[k][i]) - (s * V[k][j]); /* new ith column = c*(ith column) - s*(jth column) */
            v2[k-1] = (s * V[k][i]) + (c * V[k][j]); /* new jth column = s*(ith column) + c*(jth column) */
        }
    for (k = 1; k < n+1; k++){
        V[k][i] = v1[k-1];
        V[k][j] = v2[k-1];
    }
    free(v1);
    free(v2);
    return 0;
}

/*
The function receives the n*n sized matrix A, the indices i & j, and the values c & s.
The function multiplies A = P^T * A * P.
The matrix P is given as described in the instructions.
*/
int update_e_value_mat(double** A, double c, double s, int n, int i, int j){
    double** temp;
    int r;
    double d1, d2, offd;

    temp = (double**) calloc(2, sizeof(double*));
    if (temp == NULL) {
        return 1;
    }
    for (r = 0; r < 2; r++){
        temp[r] = (double*) calloc(n, sizeof(double));
        if (temp[r] == NULL){
            free_matrix(temp, r);
            return 1;
        }
    }
    
    d1 = (pow(c, 2) * A[i][i]) + (pow(s, 2) * A[j][j]) - (2 * c * s * A[i][j]);
    d2 = (pow(s, 2) * A[i][i]) + (pow(c, 2) * A[j][j]) + (2 * c * s * A[i][j]);
    offd = 0.0;

    for (r = 0; r < n; r++){
        if (r != i && r != j){
            temp[0][r] = (c * A[r][i]) - (s * A[r][j]);
            temp[1][r] = (c * A[r][j]) + (s * A[r][i]);
        }
    }

    for (r = 0; r < n; r++){
        if (r != i && r != j){
            A[r][i] = temp[0][r];
            A[i][r] = temp[0][r];
            A[r][j] = temp[1][r];
            A[j][r] = temp[1][r];
        }
    }
    A[i][i] = d1;
    A[j][j] = d2;
    A[i][j] = offd;
    A[j][i] = offd;
    free_matrix(temp, 2);
    return 0;
}

/*
The function receives a symmetric double matrix of size n*n.
The function uses the Jacobi iterative method to calculate the matrix' e-values and e-vectors.
The function returns the matrix V whose first row is the e-values, 
and the column below each e-value is the corresponding e-vector.
*/
double** jacobi_eval_evec(double** mat, int n){
    double** A, **V;
    int* midx;
    int i, j, iter, flag;
    double sso1, sso2, eps, s, c, theta, t;
    
    flag = 0;
    eps = 0.00001;
    iter = 100;
    A = (double**) calloc(n, sizeof(double*));
    if (A == NULL) {
        return NULL;
    }
    V = (double**) calloc(n+1, sizeof(double*));
    if (V == NULL) {
        return NULL;
    }
    for (i = 0; i < n+1; i++){
        if (i < n){
            A[i] = (double*) calloc(n, sizeof(double));
            if (A[i] == NULL) {
                free_matrix(A, i);
                return NULL;
            }
            for (j = 0; j < n; j++){
                A[i][j] = mat[i][j];
            }
        }
        V[i] = (double*) calloc(n, sizeof(double));
        if (V[i] == NULL) {
            free_matrix(V, i);
            return NULL;
        }
        if (i > 0){
            V[i][i-1] = 1.0; /* V[1:, :] = I */
        }
    }

    if (n > 1){
        sso2 = sum_off_diag_sq(A, n);
        do
        {
            sso1 = sso2;
            midx = max_abs_off_diag(A, n);
            i = midx[0];
            j = midx[1];
            free(midx);
            if (A[i][j] == 0) { break; } /* If reached a diagonal matrix, done */
            theta = (A[j][j] - A[i][i]) / (2.0 * A[i][j]);
            t = sign(theta) / (fabs(theta) + sqrt(pow(theta, 2) + 1.0));
            c = 1.0 / sqrt(pow(t, 2) + 1);
            s = t * c;
            if (update_e_value_mat(A, c, s, n, i, j) == 1){ /* A = P^T * A * P */
                flag = 1;
                break;
            }
            sso2 = sum_off_diag_sq(A, n);
            iter--;
            if (update_e_vector_mat(V, c, s, n, i, j) == 1){ /* V[1:, :] = V[1:, :] * P */
                flag = 1;
                break;
            }
        } while (((sso1 - sso2) > eps) || (iter > 0));
    }
    if (flag == 0){
        for (i = 0; i < n; i++){
            V[0][i] = A[i][i]; /* V[1, :] = DIAG(A) */
        }
    }
    free_matrix(A, n);

    if (flag == 1){
        free_matrix(V, n);
        return NULL;
    }
    return V;
}

/*
The function receives a double matrix of size n*d.
The function prints the matrix row by row with 4 digits to the right of the dot.
*/
void print_mat(double** mat, int n, int d) {
    int i, j;
    
    for (i = 0; i < n; i++) {
        for (j = 0; j < d - 1; j++) {
            printf("%.4f,", mat[i][j]);
        }
        printf("%.4f\n", mat[i][d-1]);
    }
}

/*
The function is a wrapper function that recieves a double matrix of size n*d.
The function navigates through the needed functions in order to produce the desired goal.
*/
double** calculate_mat(double** mat, char* goal ,int n, int d){
    double** wam, **ddg, **lnorm, **jacobi;

    if (strcmp(goal, "jacobi") == 0){
        jacobi = jacobi_eval_evec(mat, n);
        if (jacobi == NULL){
            return NULL;
        }
        return jacobi;
    }
    
    wam = weighted_adj_mat(mat, n, d);
    if ((wam == NULL) || (strcmp(goal, "wam") == 0)){
        return wam;
    }
    else {
        ddg = diag_deg_mat(wam, n);
        if ((ddg == NULL) || (strcmp(goal, "ddg") == 0)){
            free_matrix(wam, n);
            return ddg;
        }
        else {
            lnorm = norm_graph_lap(wam, ddg, n);
            free_matrix(wam, n);
            free_matrix(ddg, n);
            return lnorm;
        }
    }
}

int main(int argc, char *argv[]){
    char* goal;
    char* input_file_path;
    double** obs, **result;
    int dims[2];
    int jacobi_flag;

    if (first_input_validation(argc, argv) == 1){
        return 1;
    }
    
    goal = argv[1];
    input_file_path = argv[2];

    if (find_dimensions(input_file_path, dims) == 1){
        return 1;
    }

    obs = read_file(input_file_path, dims[0], dims[1]);
    if (obs == NULL){
        return 1;
    }

    result = calculate_mat(obs, goal, dims[0], dims[1]);
    free_matrix(obs, dims[0]);
    if (result == NULL){
        printf("An Error Has Occurred\n");
        return 1;
    }
    jacobi_flag = strcmp(goal, "jacobi") == 0;  /* A boolean flag for jacobi specific use */
    print_mat(result, dims[0] + jacobi_flag, dims[0]);
    free_matrix(result, dims[0] + jacobi_flag);
    return 0;
}