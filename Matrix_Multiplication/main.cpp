#include <bits/stdc++.h>
using namespace std;

int row_A, col_A, col_B; 
int** mat_A;
int** mat_B;
int** mat_C;

void mul_serial(){
    for(int i = 0; i < row_A; i++){
        for(int j = 0; j < col_B; j++){
            for(int k = 0; k < col_A; k++){
                // printf("%d x %d, ", mat_A[i][k], mat_B[k][j]);
                mat_C[i][j] += mat_A[i][k] * mat_B[k][j];
            }
            // printf("\n");
        }
    }
}

int** init(int row, int col, bool is_C){
    int** mat = (int**) malloc(row * sizeof(int *));
    for(int i = 0; i < row; i++){
        mat[i] = (int *) malloc(col * sizeof(int));
    }

    random_device rd;
    mt19937 generator(rd());
    uniform_int_distribution<int> unif(INT_MIN, INT_MAX);

    for(int i = 0; i < row; i++){
        for(int j = 0; j < col; j++){
            // mat_A[i][j] = mat_B[i][j]= unif(generator);
            mat[i][j] = is_C ? 0 : i * col + j;
        }
    }

    return mat;
}

int main(int argc, char* argv[]){
    if(argc != 4){
        cerr << "Usage: ./a.out $row_A $col_A $col_B " << '\n';
        exit(-1);
    }
    row_A = atoi(argv[1]);
    col_A = atoi(argv[2]);
    col_B = atoi(argv[3]);
    assert(row_A > 0 && col_A > 0 && col_B > 0);

    mat_A = init(row_A, col_A, false);
    mat_B = init(col_A, col_B, false);
    mat_C = init(row_A, col_B, true);

    mul_serial();
    
    for(int i = 0; i < row_A; i++){
        for(int j = 0; j < col_B; j++){
            printf("%d ", mat_C[i][j]);
        }
        printf("\n");
    }

    return 0;
}
