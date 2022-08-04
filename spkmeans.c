#define sign(x) (x < 0 ? -1 : 1)
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <math.h>
#include <string.h>

/*
The function check if the input is of the right length.
Then, the function checks whether goal is valid.
*/
int first_input_validation(int length_of_input, char *input[])
{
    if (length_of_input != 3){
        printf("Invalid Input!");
        return 1;
    }
    if (strcmp(input[1], "wam") != 0 && strcmp(input[1], "ddg") != 0 && strcmp(input[1], "lnorm") != 0 &&
    strcmp(input[1], "") != 0){
        printf("Invalid Input!");
        return 1;
    }
    return 0;
}

/*
The function retrieves the dimension of the input file.
The function inputs said dimensions (# of rows and columns) into the dims array.
*/
int find_dimensions(char const *filename, int *dims){
    FILE *f = NULL;
    char c;

    f = fopen(filename, "r");
    if (f == NULL) {
        fclose(f);
        printf("An Error Has Occurred");
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
The function returns a double matrix of size rows * columns whose elements are that of the input file.
*/
double** read_file(char const *filename, int rows, int columns) 
{
    FILE *f = NULL;
    char c;
    int i, j;
    double** obs;

    f = fopen(filename, "r");
    if (f == NULL) {
        fclose(f);
        printf("An Error Has Occurred");
        return NULL;
    }
    obs = calloc(rows, sizeof(double*));
    for (i = 0; i < rows; i++)
    {
        obs[i] = calloc(columns, sizeof(double));
        for (j = 0; j < columns; j++)
        {
            fscanf(f, "%lf%c", &obs[i][j], &c);
        }
    }
    fclose(f);
    return obs;
}

/*
The function returns the euclidean distance between the two double vectors.
The function assumes the dimension of the vectors is columns.
*/
double euclid_dist(double* x, double* y, int columns){
    double dist;
    int j;

    dist = 0;
    for (j = 0; j < columns; j++){
        dist += (x[j]-y[j]) * (x[j]-y[j]);
    }
    return sqrt(dist);
}

double** weighted_adj_mat(double** obs, int n, int d){
    double** wam;
    int i, j;

    wam = calloc(n, sizeof(double*));
    for (i = 0; i < n; i++){
        wam[i] = calloc(n, sizeof(double));
        wam[i][i] = 0;
        for (j = i + 1; j < n; j++){
            wam[i][j] = exp(- euclid_dist(obs[i], obs[j], d) / 2);
            wam[j][i] = wam[i][j];
        }
    }
    return wam;
}

double** diag_deg_mat(double** wam, int n){
    double** ddg;
    int i, j;

    ddg = calloc(n, sizeof(double*));
    for (i = 0; i < n; i++){
        ddg[i] = calloc(n, sizeof(double));
        for (j = 0; j < n; j++){
            ddg[i][i] += wam[i][j];
        }
    }
    return ddg;
}

double** norm_graph_lap(double** wam, double** ddg, int n){
    double** lnorm;
    int i, j;

    for (i = 0; i < n; i++){
        ddg[i][i] = 1 / sqrt(ddg[i][i]);
    }
    
    lnorm = calloc(n, sizeof(double*));
    for (i = 0; i < n; i++){
        lnorm[i] = calloc(n, sizeof(double));
        lnorm[i][i] = 1;
        for (j = i + 1; j < n; j++){
            lnorm[i][j] = - wam[i][j] * ddg[i][i] * ddg[j][j];
            lnorm[j][i] = lnorm[i][j];
        }
    }
    for (i = 0; i < n; i++){ /* Consider deleting! */
        ddg[i][i] = 1 / (ddg[i][i] * ddg[i][i]);
    }
    return lnorm;
}

int* max_abs_off_diag(double** mat, int n){
    int i, j;
    int *max_indices;
    max_indices = calloc(2, sizeof(int));
    max_indices[0] = 0;
    max_indices[1] = 1;
    for (i = 0; i < n; i++){
        for (j = i + 1; j < n; j++){
            if (fabs(mat[max_indices[0]][max_indices[1]]) < fabs(mat[i][j])){
                max_indices[0] = i;
                max_indices[1] = j;
            }
        }
    }
    return max_indices;
}

double sum_off_diag_sq(double** mat, int n){
    int i, j;
    double s = 0;

    for (i = 0; i < n; i++){
        for (j = i+1; j < n; j++){
            s += 2 * mat[i][j] * mat[i][j];
        }
    }
    return s;
}

void update_e_vector_mat(double** V, double c, double s, int n, int i, int j){
    double* v1;
    double* v2;
    int k;
    v1 = calloc(n, sizeof(double));
    v2 = calloc(n, sizeof(double));
    
    for (k = 1; k < n+1; k++){
            v1[k-1] = c * V[k][i] - s * V[k][j];
            v2[k-1] = s * V[k][i] + c * V[k][j];
        }
    for (k = 1; k < n+1; k++){
        V[k][i] = v1[k-1];
        V[k][j] = v2[k-1];
    }
    
    free(v1);
    free(v2);
}

void free_matrix(double** mat, int n) {
    int i;
    if (mat != NULL) {
        for (i = 0; i < n; i++) {
            free(mat[i]);
        }
        free(mat);
    }
}

void update_e_value_mat(double** A, double c, double s, int n, int i, int j){
    double** temp;
    int r;
    double d1, d2, offd;

    temp = calloc(2, sizeof(double*));
    temp[0] = calloc(n, sizeof(double));
    temp[1] = calloc(n, sizeof(double));
    
    d1 = c * c * A[i][i] + s * s * A[j][j] - 2 * c * s * A[i][j];
    d2 = s * s * A[i][i] + c * c * A[j][j] + 2 * c * s * A[i][j];
    offd = 0;

    for (r = 0; r < n; r++){
        if (r != i && r != j){
            temp[0][r] = c * A[r][i] - s * A[r][j];
            temp[1][r] = c * A[r][j] + s * A[r][i];
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
}

double** jacobi_eval_evec(double** mat, int n){
    double** A;
    double** V;
    int* midx;
    int i, j, iter;
    double sso1, sso2, eps, s, c, theta, t;
    
    eps = 0.00001;
    iter = 100;
    A = calloc(n, sizeof(double*));
    V = calloc(n+1, sizeof(double*));
    for (i = 0; i < n; i++){
        A[i] = calloc(n, sizeof(double));
        for (j = 0; j < n; j++){
            A[i][j] = mat[i][j];
        }
        V[i+1] = calloc(n, sizeof(double));
        V[i+1][i] = 1;
    }
    V[0] = calloc(n, sizeof(double));
    
    sso2 = sum_off_diag_sq(A, n);
    do
    {
        sso1 = sso2;
        midx = max_abs_off_diag(A, n);
        theta = (A[midx[1]][midx[1]] - A[midx[0]][midx[0]]) / (2 * A[midx[0]][midx[1]]);
        t = sign(theta) / (fabs(theta) + sqrt(theta * theta + 1));
        c = 1 / sqrt(t * t + 1);
        s = t * c;
        update_e_value_mat(A, c, s, n, midx[0], midx[1]);
        sso2 = sum_off_diag_sq(A, n);
        iter--;
        update_e_vector_mat(V, c, s, n, midx[0], midx[1]);
    } while (sso1 - sso2 > eps || iter > 0);
    
    for (i = 0; i < n; i++){
        V[0][i] = A[i][i];
    }
    
    free(midx);
    free_matrix(A, n);

    return V;
}

void print_mat(double** mat, int n, int m) {
    int i, j;
    for (i = 0; i < n; i++) {
        for (j = 0; j < m - 1; j++) {
            printf("%.4f,", mat[i][j]);
        }
        printf("%.4f\n", mat[i][m-1]);
    }
}

int main(int argc, char *argv[]){
    char* goal;
    char* input_file_path;
    double** obs;
    double** wam;
    double** ddg;
    double** lnorm;
    double** jacobi;
    int dims[2];

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

    wam = weighted_adj_mat(obs, dims[0], dims[1]);

    if (strcmp(goal, "wam") == 0) {
        print_mat(wam, dims[0], dims[0]);
    }

    else {
        ddg = diag_deg_mat(wam, dims[0]);
        if (strcmp(goal, "ddg") == 0) {
            print_mat(ddg, dims[0], dims[0]);
        }

        else {
            lnorm = norm_graph_lap(wam, ddg, dims[0]);
            if (strcmp(goal, "lnorm") == 0) {
                print_mat(lnorm, dims[0], dims[0]);
            }

            else {
                jacobi = jacobi_eval_evec(lnorm, dims[0]);
                print_mat(jacobi, dims[0] + 1, dims[0]);
            }   
        }
    }
    free_matrix(wam, dims[0]);
    free_matrix(ddg, dims[0]);
    free_matrix(lnorm, dims[0]);
    free_matrix(jacobi, dims[0] + 1);
    return 0;
}