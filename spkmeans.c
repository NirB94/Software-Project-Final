#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <math.h>

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
    strcmp(input[1], "jacobi") != 0){
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
    for (i = 0; i < n; i++){
        ddg[i][i] = 1 / (ddg[i][i] * ddg[i][i]);
    }
    return lnorm;
}

int** max_abs_off_diag(double** mat, int n){
    int i, j;
    int max_indices[2] = {0, 1};
    for (i = 0; i < n; i++){
        for (j = i + 1; j < n; j++){
            if (abs(mat[max_indices[0]][max_indices[1]] < abs(mat[i][j]))){
                max_indices[0] = i;
                max_indices[1] = j;
            }
        }
    }
    return max_indices;
}

double** jacobi(double** mat, int n){
    
}


int main(int argc, char *argv[]){
    char* goal;
    char* input_file_path;
    double** obs;
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
}