#include <stdlib.h>
#include <iostream>
#include "util.h"

Convolution::Convolution(int m, int c, int r, int s, int sx, int sy, int px, int py)
{
  M = m;
  C = c;
  R = r;
  S = s;
  Sx = sx;
  Sy = sy;
  Px = px;
  Py = py;
  weights = (DATA*) malloc(M * C * R * S * sizeof(DATA));
  DATA (*temp)[C][R][S] = (DATA (*)[C][R][S])weights;
  for(int i=0; i<M; i++)
    for(int j=0; j<C; j++)
      for(int k=0; k<R; k++)
        for(int l=0; l<S; l++)
          temp[i][j][k][l] = (i*C*R*S+j*R*S+k*S+l)%256;
}

Linear::Linear(int m, int l)
{
  M = m;
  L = l;
  weights = (DATA*) malloc(M * L * sizeof(DATA));
  DATA (*temp)[L] = (DATA (*)[L])weights;
  for(int i=0; i<M; i++)
    for(int j=0; j<L; j++)
      temp[i][j] = (i*L+j)%256;
}
fmap* Convolution::conv_2d(fmap* input_features)
{
  fmap* output_features_pointer; fmap output_features;

  DATA* temp_output;

  int* start_pointer;
  int* temp_input; int* temp_in_prel;
  int32_t* temp_vect;

  clock_t start,end;

  int32_t temp_input_vect[8], temp_weight_vect[8];
  int temp_vect_int[8];
  int horizontal_stride;

  __m256 input_vector, weight_vector, product_vector, output_vector;
  __m256i input_vector_int, weight_vector_int, output_vector_int;

  output_features.dim1 = input_features->dim1;
  output_features.dim2 = M;
  output_features.dim4 = (input_features->dim4-S+2*Px)/Sx + 1; //Check for width or padding waiver
  output_features.dim3 = (input_features->dim3-R+2*Py)/Sy + 1; //Check for height

  temp_input = (int*) malloc(input_features->dim1 * input_features->dim2 * (input_features->dim3 + 2*Py) * (input_features->dim4 + 2*Px) * sizeof(int));
  output_features.data = (DATA*) malloc(output_features.dim1 * output_features.dim2 * output_features.dim3 * output_features.dim4 * sizeof(DATA));

  for(int l=0; l<input_features->dim1; l++)                                                                   //Data re-arrangement for padding
    for(int k=0; k<input_features->dim2; k++)
      for(int j=0; j<(input_features->dim3+2*Py); j++)
        for(int i=0; i<(input_features->dim4+2*Px); i++){
          if(i<Px || ((input_features->dim4+2*Px)-i-1)<Px || j<Py || ((input_features->dim3+2*Py)-j-1)<Py){}
          else {
            *(temp_input + l*(input_features->dim3+2*Py)*input_features->dim2*(input_features->dim4+2*Px) + k*(input_features->dim3+2*Py)*(input_features->dim4+2*Px) + j*(input_features->dim4+2*Px) + i) = *(input_features->data + l*input_features->dim3*input_features->dim2*input_features->dim4 + k*input_features->dim4*input_features->dim3 + (j-Py)*input_features->dim4 + (i-Px)); }

}
  delete [] input_features->data;

  temp_in_prel = temp_input;
  temp_output = output_features.data;

  horizontal_stride = (S/8 < 1)?1:(S/8);

  start = clock(); 

  for(int n=0; n<output_features.dim1; n++){         //Output Batch loop
    for(int m=0; m<output_features.dim2; m++){       //Output Channel loop
      temp_input = temp_in_prel + n*(input_features->dim3 + (2*Py))*(input_features->dim4 + (2*Px))*input_features->dim2;
      for(int l=0; l<output_features.dim3; l++){     //Output Height loop
       for(int k=0; k<output_features.dim4; k++){    //Output Width loop
       //Iteration for ouptut channels and values done

         start_pointer = (int*)temp_input;           //Cast from int32_t to int pointer

         output_vector = _mm256_setr_ps(0,0,0,0,0,0,0,0);                           //Setting all registers to zero

         for(int p=0; p<input_features->dim2; p++){               //Input Channel loop
           for(int j=0; j < R; j++){                              //Input Height loop
             for(int i=0; i<horizontal_stride; i++){              //Input Width loop

               if(S >= 8){  
                 input_vector_int = _mm256_setr_epi32(*(start_pointer + (i*8) + 0),*(start_pointer + (i*8) + 1), *(start_pointer + (i*8) + 2), *(start_pointer + (i*8) + 3), *(start_pointer + (i*8) + 4), *(start_pointer + (i*8) + 5), *(start_pointer + (i*8) + 6), *(start_pointer + (i*8) + 7)); 
                 input_vector  = _mm256_cvtepi32_ps(input_vector_int);

                 weight_vector_int = _mm256_setr_epi32(*(weights + (m*R*S) + (j*S) + (i*8) +0),*(weights + (m*R*S) + (j*S) + (i*8) +1),*(weights + (m*R*S) + (j*S) + (i*8) +2),*(weights + (m*R*S) + (j*S) + (i*8) +3),*(weights + (m*R*S) + (j*S) + (i*8) +4),*(weights + (m*R*S) + (j*S) + (i*8) +5),*(weights + (m*R*S) + (j*S) + (i*8) +6),*(weights + (m*R*S) + (j*S) + (i*8) + 7));
                 weight_vector = _mm256_cvtepi32_ps(weight_vector_int);

                 product_vector = _mm256_mul_ps(input_vector,weight_vector);
                 output_vector = _mm256_add_ps(output_vector,product_vector);

               }

               if(((S > (i+1)*8) && (i==(S/8)-1)) || (S/8 < 1)){ 
                 for(int q=0; q<8; q++){
                   if(S/8 > 0){
                     if(q < S-((i+1)*8)){  
                     temp_input_vect[q]  = *(start_pointer + ((i+1)*8) + q); 
                     temp_weight_vect[q] = *(weights + (m*R*S) + (j*S) + ((i+1)*8) + q);
                     }
                     else{
                       temp_input_vect[q]  = 0;
                       temp_weight_vect[q] = 0;
                     }
                   }
                   else{
                     if(q < S){  
                       temp_input_vect[q]  = *(start_pointer + q); 
                       temp_weight_vect[q] = *(weights + (m*R*S) + (j*S) + q);
                     }
                     else{
                       temp_input_vect[q]  = 0;
                       temp_weight_vect[q] = 0;
                     }
                   }
                 }

                 input_vector_int = _mm256_setr_epi32(temp_input_vect[0],temp_input_vect[1],temp_input_vect[2],temp_input_vect[3],temp_input_vect[4],temp_input_vect[5],temp_input_vect[6],temp_input_vect[7]);
                 weight_vector_int = _mm256_setr_epi32(temp_weight_vect[0],temp_weight_vect[1],temp_weight_vect[2],temp_weight_vect[3],temp_weight_vect[4],temp_weight_vect[5],temp_weight_vect[6],temp_weight_vect[7]);
                 
                 input_vector  = _mm256_cvtepi32_ps(input_vector_int);
                 weight_vector = _mm256_cvtepi32_ps(weight_vector_int);

                 product_vector = _mm256_mul_ps(input_vector,weight_vector);
                 output_vector = _mm256_add_ps(output_vector,product_vector);
               }
             }
             start_pointer = start_pointer + (input_features->dim4 + (2*Px)); 
           }  
           start_pointer = temp_input + (p+1)*((input_features->dim3 + (2*Py))*(input_features->dim4 + (2*Px))); 
         }

         output_vector_int = _mm256_cvtps_epi32(output_vector);
          
         _mm256_storeu_si256((__m256i*)&temp_vect_int[0],output_vector_int);

         temp_vect = (int32_t*)temp_vect_int;

         temp_output[(n*output_features.dim3*output_features.dim2*output_features.dim4) + (m*output_features.dim3*output_features.dim4) + (l*output_features.dim4)+k] = temp_vect[0] + temp_vect[1] + temp_vect[2] + temp_vect[3] + temp_vect[4] + temp_vect[5] + temp_vect[6] + temp_vect[7];

         temp_input = temp_in_prel + n*(input_features->dim3 + (2*Py))*(input_features->dim4 + (2*Px))*input_features->dim2 + l*(input_features->dim4 + (2*Px))*Sy + ((k+1)*Sx);
        }
       }
    }
  }
 
  end = clock();

  exec_time = double(end-start) / double(CLOCKS_PER_SEC);
 
  delete [] temp_in_prel;

  output_features_pointer = &output_features;

  return output_features_pointer;
}

fmap* Convolution::conv2d_IS(fmap* input_features)                                 //Ensures a input vector is re-used to maximum, re-use within vectors isn't supported
{
  fmap* output_features_pointer; fmap output_features;

  __m256i weight_vector_int, input_vector_int, product_vector;
  __m256  input_vector_float, product_vector_float, weight_vector_float;

  int temp[8];
  int t, index, output_index, horizontal_stride, n, c, h, w;
  int* temp_in_init;

  clock_t start,end;

  output_features.dim1 = n = input_features->dim1;
  output_features.dim2 = c = M;
  output_features.dim4 = w = (input_features->dim4-S+2*Px)/Sx + 1;
  output_features.dim3 = h = (input_features->dim3-R+2*Py)/Sy + 1;

  output_features.data = (DATA*) malloc(n * c * h * w * sizeof(DATA));
  temp_in_init = (int*) malloc(input_features->dim1 * input_features->dim2 * (input_features->dim3 + 2*Py) * (input_features->dim4 + 2*Px) * sizeof(int));

  DATA (*temp_out)[c][h][w]        = (DATA (*)[c][h][w])output_features.data;
  int (*temp_in_prel)[input_features->dim2][input_features->dim3][input_features->dim4]     = (DATA (*)[input_features->dim2][input_features->dim3][input_features->dim4])input_features->data;
  int (*temp_in) [input_features->dim2][input_features->dim3+(2*Py)][input_features->dim4+(2*Px)] = (int (*)[input_features->dim2][input_features->dim3+(2*Py)][input_features->dim4+(2*Px)])temp_in_init;
  int (*temp_weight)[C][R][S]      = (DATA (*)[C][R][S])weights;

  for(int l=0; l<input_features->dim1; l++)                                                                   //Data re-arrangement for padding
    for(int k=0; k<input_features->dim2; k++)
      for(int j=0; j<(input_features->dim3+2*Py); j++)
        for(int i=0; i<(input_features->dim4+2*Px); i++) 
          if(i<Px || ((input_features->dim4+2*Px)-i-1)<Px || j<Py || ((input_features->dim3+2*Py)-j-1)<Py)
            temp_in[l][k][j][i] = 0; 
          else
            temp_in[l][k][j][i] = temp_in_prel[l][k][j-Py][i-Px];

  delete [] input_features->data;

  horizontal_stride = (S/8) + 1;

  start = clock();

  for(int q=0; q<n; q++){                     //Batch size loop
   for(int u=0; u<c; u++){                    //Number of output channels loop/number of filters loop
    for(int p=0; p<w; p++ ){                  //Number of outpupt colunmns loop

      for(int m=0; m<C; m++){

        for(int l=0; l<horizontal_stride; l++){

          for(int k=0; k<input_features->dim3; k++){
               
            if(S>8 && l==0){
              input_vector_int   = _mm256_setr_epi32(temp_in[q][m][k][p*Sx],temp_in[q][m][k][p*Sx+1],temp_in[q][m][k][p*Sx+2],temp_in[q][m][k][p*Sx+3],temp_in[q][m][k][p*Sx+4],temp_in[q][m][k][p*Sx+5],temp_in[q][m][k][p*Sx+6],temp_in[q][m][k][p*Sx+7]);                                                                 //Load input only once per column
            }
            else if(S<8){
              for(int i=0; i<8; i++){
                temp[i] = (i<S)?temp_in[q][m][k][p*Sx+i]:0;
              }
              input_vector_int   = _mm256_setr_epi32(temp[0],temp[1],temp[2],temp[3],temp[4],temp[5],temp[6],temp[7]);
            }
            else if (S>8 && l==1){
              for(int i=0; i<8; i++){
                temp[i] = (i<(S-8))?temp_in[q][m][k][p*Sx+i]:0;
              }
              input_vector_int   = _mm256_setr_epi32(temp[0],temp[1],temp[2],temp[3],temp[4],temp[5],temp[6],temp[7]);
            }

            input_vector_float = _mm256_cvtepi32_ps(input_vector_int);
            
            if(k<R){
              t = k/Sy;

              index = k%R;
              for(int i=0;i<(t+1);i++){
                weight_vector_int   = _mm256_setr_epi32(temp_weight[u][m][index-(i*Sy)][0],temp_weight[u][m][index-(i*Sy)][1],temp_weight[u][m][index-(i*Sy)][2],temp_weight[u][m][index-(i*Sy)][3],temp_weight[u][m][index-(i*Sy)][4],temp_weight[u][m][index-(i*Sy)][5],temp_weight[u][m][index-(i*Sy)][6],temp_weight[u][m][index-(i*Sy)][7]);   //Load weights
                weight_vector_float = _mm256_cvtepi32_ps(weight_vector_int);                                     

                product_vector_float = _mm256_mul_ps(input_vector_float,weight_vector_float);                      //Multiply the vectors

                product_vector = _mm256_cvtps_epi32(product_vector_float);                                         //Convert float to integer
                _mm256_storeu_si256((__m256i*)&temp[0],product_vector);                                            //Store in memory
            
                temp_out[q][u][i][p] = temp_out[q][u][i][p] + temp[7] + temp[6] + temp[5] + temp[4] + temp[3] + temp[2] + temp[1] + temp[0];  //Local addition within vector
                
              }
            }
            else if(k<(input_features->dim3-R)){
              index = k - (k/Sy)*Sy;

              if(k<((t+1)*Sy) || k>(input_features->dim3-(t+1)*Sy)){   //Input vectors for which number of reuse is less due to bigger strides 
                for(int i=0; i<t; i++){
                  output_index = (k/Sy) - i;

                  weight_vector_int   = _mm256_setr_epi32(temp_weight[u][m][index+(i*Sy)][0],temp_weight[u][m][index+(i*Sy)][1],temp_weight[u][m][index+(i*Sy)][2],temp_weight[u][m][index+(i*Sy)][3],temp_weight[u][m][index+(i*Sy)][4],temp_weight[u][m][index+(i*Sy)][5],temp_weight[u][m][index+(i*Sy)][6],temp_weight[u][m][index+(i*Sy)][7]);
                  weight_vector_float = _mm256_cvtepi32_ps(weight_vector_int);

                  product_vector_float = _mm256_mul_ps(input_vector_float,weight_vector_float);

                  product_vector = _mm256_cvtps_epi32(product_vector_float);
                  _mm256_storeu_si256((__m256i*)&temp[0],product_vector);

                  temp_out[q][u][output_index][p] = temp_out[q][u][output_index][p] + temp[7] + temp[6] + temp[5] + temp[4] + temp[3] + temp[2] + temp[1] + temp[0];
                }
              }
              else{
                for(int i=0; i<(t+1); i++){
                  output_index = (k/Sy) - i;

                  weight_vector_int   = _mm256_setr_epi32(temp_weight[u][m][index+(i*Sy)][0],temp_weight[u][m][index+(i*Sy)][1],temp_weight[u][m][index+(i*Sy)][2],temp_weight[u][m][index+(i*Sy)][3],temp_weight[u][m][index+(i*Sy)][4],temp_weight[u][m][index+(i*Sy)][5],temp_weight[u][m][index+(i*Sy)][6],temp_weight[u][m][index+(i*Sy)][7]);
                  weight_vector_float = _mm256_cvtepi32_ps(weight_vector_int);

                  product_vector_float = _mm256_mul_ps(input_vector_float,weight_vector_float);

                  product_vector = _mm256_cvtps_epi32(product_vector_float);
                  _mm256_storeu_si256((__m256i*)&temp[0],product_vector);

                  temp_out[q][u][output_index][p] = temp_out[q][u][output_index][p] + temp[7] + temp[6] + temp[5] + temp[4] + temp[3] + temp[2] + temp[1] + temp[0];
                }
              }
            }
            else{
              index = R - (input_features->dim3 - k - 1) % R - 1;

              t = ((input_features->dim3-k)%R)/Sy;

              for(int i=0; i<(t+1); i++){
                output_index = (k/Sy) - i;

                weight_vector_int   = _mm256_setr_epi32(temp_weight[u][m][index+(i*Sy)][0],temp_weight[u][m][index+(i*Sy)][1],temp_weight[u][m][index+(i*Sy)][2],temp_weight[u][m][index+(i*Sy)][3],temp_weight[u][m][index+(i*Sy)][4],temp_weight[u][m][index+(i*Sy)][5],temp_weight[u][m][index+(i*Sy)][6],temp_weight[u][m][index+(i*Sy)][7]);
                weight_vector_float = _mm256_cvtepi32_ps(weight_vector_int);

                 product_vector_float = _mm256_mul_ps(input_vector_float,weight_vector_float);

                 product_vector = _mm256_cvtps_epi32(product_vector_float);
                 _mm256_storeu_si256((__m256i*)&temp[0],product_vector);

                 temp_out[q][u][output_index][p] = temp_out[q][u][output_index][p] + temp[7] + temp[6] + temp[5] + temp[4] + temp[3] + temp[2] + temp[1] + temp[0];
              }
            }
          }
        }
      }
    }
   }
  }

  end = clock();
 
  exec_time = double(end-start) / double(CLOCKS_PER_SEC);

  delete [] temp_in;

  output_features_pointer = &output_features;

  return output_features_pointer;
}

fmap* Convolution::conv2d_OS(fmap* input_features)
{
  fmap* output_features_pointer; fmap output_features;

  __m256i weight_vector_int, input_vector_int, mac_vector;
  __m256  input_vector_float, product_vector_float, weight_vector_float, mac_vector_float;

  int temp_weight_static[8], temp_input[8], temp[8];
  int horizontal_stride,n,c,h,w;

  clock_t start,end;

  int* temp_in_init;

  output_features.dim1 = n = input_features->dim1;
  output_features.dim2 = c = M;
  output_features.dim3 = h = (input_features->dim3-S+2*Py)/Sy + 1;
  output_features.dim4 = w = (input_features->dim4-R+2*Px)/Sx + 1;

  output_features.data = (DATA*) malloc(n * c * h * w * sizeof(DATA));
  temp_in_init = (int*) malloc(input_features->dim1 * input_features->dim2 * (input_features->dim3 + 2*Py) * (input_features->dim4 + 2*Px) * sizeof(int));

  DATA(*temp_out)[c][h][w]        = (DATA (*)[c][h][w])output_features.data;
  int (*temp_in_prel)[input_features->dim2][input_features->dim3][input_features->dim4]     = (DATA (*)[input_features->dim2][input_features->dim3][input_features->dim4])input_features->data;
  int (*temp_in) [input_features->dim2][input_features->dim3+(2*Py)][input_features->dim4+(2*Px)] = (int (*)[input_features->dim2][input_features->dim3+(2*Py)][input_features->dim4+(2*Px)])temp_in_init;
  int (*temp_weight)[C][R][S]      = (DATA (*)[C][R][S])weights;

  for(int l=0; l<input_features->dim1; l++)                                                                   //Data re-arrangement for padding
    for(int k=0; k<input_features->dim2; k++)
      for(int j=0; j<(input_features->dim3+2*Py); j++)
        for(int i=0; i<(input_features->dim4+2*Px); i++) 
          if(i<Px || ((input_features->dim4+2*Px)-i-1)<Px || j<Py || ((input_features->dim3+2*Py)-j-1)<Py)
            temp_in[l][k][j][i] = 0; 
          else
            temp_in[l][k][j][i] = temp_in_prel[l][k][j-Py][i-Px];

  delete [] input_features->data;

  horizontal_stride = (S/8) + 1;

  start = clock();

  for(int q=0; q<n; q++){
    for(int p=0; p<c; p++){
      for(int m=0; m<h; m++){
        for(int l=0; l<w; l++){

          mac_vector_float = _mm256_setr_ps(0,0,0,0,0,0,0,0);

          for(int k=0; k<C; k++){
            for(int j=0; j<horizontal_stride; j++){
              for(int i=0; i<R; i++){

                if(S>8 && j==0){
                  input_vector_int   = _mm256_setr_epi32(temp_in[q][k][m*Sy + i][l*Sx],temp_in[q][k][m*Sy + i][l*Sx+1],temp_in[q][k][m*Sy + i][l*Sx+2],temp_in[q][k][m*Sy + i][l*Sx+3],temp_in[q][k][m*Sy + i][l*Sx+4],temp_in[q][k][m*Sy + i][l*Sx+5],temp_in[q][k][m*Sy + i][l*Sx+6],temp_in[q][k][m*Sy + i][l*Sx+7]);                                                                 //Load input
                  weight_vector_int  = _mm256_setr_epi32(temp_weight[p][k][i][j],temp_weight[p][k][i][j+1],temp_weight[p][k][i][j+2],temp_weight[p][k][i][j+3],temp_weight[p][k][i][j+4],temp_weight[p][k][i][j+5],temp_weight[p][k][i][j+6],temp_weight[p][k][i][j+7]);                                                                                 //Load weight

                }
                else if(S<8){
                  for(int z=0; z<8; z++){
                    temp_input[z] = (z<S)?temp_in[q][k][m*Sy + i][l*Sx+z]:0;
                    temp_weight_static[z] = (z<S)?temp_weight[p][k][i][j+z]:0;
                  }
                  input_vector_int   = _mm256_setr_epi32(temp_input[0],temp_input[1],temp_input[2],temp_input[3],temp_input[4],temp_input[5],temp_input[6],temp_input[7]);
                  weight_vector_int  = _mm256_setr_epi32(temp_weight_static[0],temp_weight_static[1],temp_weight_static[2],temp_weight_static[3],temp_weight_static[4],temp_weight_static[5],temp_weight_static[6],temp_weight_static[7]);
                }
                else if (S>8 && j==1){
                  for(int z=0; z<8; z++){
                    temp_input[z] = (z<(S-8))?temp_in[q][k][m*Sy + i][l*Sx+z]:0;
                    temp_weight_static[z] = (z<(S-8))?temp_weight[p][k][i][8+z]:0;
                  }
                  input_vector_int   = _mm256_setr_epi32(temp_input[0],temp_input[1],temp_input[2],temp_input[3],temp_input[4],temp_input[5],temp_input[6],temp_input[7]);
                  weight_vector_int  = _mm256_setr_epi32(temp_weight_static[0],temp_weight_static[1],temp_weight_static[2],temp_weight_static[3],temp_weight_static[4],temp_weight_static[5],temp_weight_static[6],temp_weight_static[7]);
                }

                input_vector_float = _mm256_cvtepi32_ps(input_vector_int); 
                weight_vector_float = _mm256_cvtepi32_ps(weight_vector_int);

                product_vector_float = _mm256_mul_ps(input_vector_float,weight_vector_float);
                mac_vector_float = _mm256_add_ps(product_vector_float,mac_vector_float);

                mac_vector = _mm256_cvtps_epi32(mac_vector_float);
          _mm256_storeu_si256((__m256i*)&temp[0],mac_vector);
              }
            }
          }
          mac_vector = _mm256_cvtps_epi32(mac_vector_float);
          _mm256_storeu_si256((__m256i*)&temp[0],mac_vector);

          temp_out[q][p][m][l] = temp[7] + temp[6] + temp[5] + temp[4] + temp[3] + temp[2] + temp[1] + temp[0];
        }
      }
    }
  }

  end = clock();

  exec_time = double(end-start) / double(CLOCKS_PER_SEC);
 
  delete [] temp_in;            

  output_features_pointer = &output_features;

  return output_features_pointer;
}

fmap* Convolution::conv2d_WS(fmap* input_features)
{
  fmap* output_features_pointer; fmap output_features;

  __m256i weight_vector_int, input_vector_int, product_vector_int;
  __m256  input_vector_float, product_vector_float, weight_vector_float;

  int temp_weight_static[8], temp_input[8], temp[8];
  int horizontal_stride,n,c,h,w;
  int* temp_in_init;

  clock_t start,end;

  output_features.dim1 = n = input_features->dim1;
  output_features.dim2 = c = M;
  output_features.dim4 = w = (input_features->dim4-S+2*Px)/Sx + 1;
  output_features.dim3 = h = (input_features->dim3-R+2*Py)/Sy + 1;

  temp_in_init = (int*) malloc(input_features->dim1 * input_features->dim2 * (input_features->dim3 + 2*Py) * (input_features->dim4 + 2*Px) * sizeof(int));
  output_features.data = (DATA*) malloc(n * c * h * w * sizeof(DATA));

  DATA (*temp_out)[c][h][w]        = (DATA (*)[c][h][w])output_features.data;
  int (*temp_in_prel)[input_features->dim2][input_features->dim3][input_features->dim4]     = (DATA (*)[input_features->dim2][input_features->dim3][input_features->dim4])input_features->data;
  int (*temp_in) [input_features->dim2][input_features->dim3+(2*Py)][input_features->dim4+(2*Px)] = (int (*)[input_features->dim2][input_features->dim3+(2*Py)][input_features->dim4+(2*Px)])temp_in_init;
  int (*temp_weight)[C][R][S]      = (DATA (*)[C][R][S])weights;

  for(int l=0; l<input_features->dim1; l++)                                                                   //Data re-arrangement for padding
    for(int k=0; k<input_features->dim2; k++)
      for(int j=0; j<(input_features->dim3+2*Py); j++)
        for(int i=0; i<(input_features->dim4+2*Px); i++)
          if(i<Px || ((input_features->dim4+2*Px)-i-1)<Px || j<Py || ((input_features->dim3+2*Py)-j-1)<Py)
            temp_in[l][k][j][i] = 0;
          else
            temp_in[l][k][j][i] = temp_in_prel[l][k][j-Py][i-Px];

  delete [] input_features->data;

  horizontal_stride = (S/8) + 1;

  start = clock();

  for(int s=0; s<M; s++){
   for(int m=0; m<C; m++){
    for(int l=0; l<R; l++){
     for(int k=0; k<horizontal_stride; k++){
 
       if(S>8 && k==0){
         weight_vector_int = _mm256_setr_epi32(temp_weight[s][m][l][k],temp_weight[s][m][l][k+1],temp_weight[s][m][l][k+2],temp_weight[s][m][l][k+3],temp_weight[s][m][l][k+4],temp_weight[s][m][l][k+5],temp_weight[s][m][l][k+6],temp_weight[s][m][l][k+7]);                                                                                 //Load weight

       }
       else if(S<8){
         for(int z=0; z<8; z++){
           temp_weight_static[z] = (z<S)?temp_weight[s][m][l][k+z]:0; 
         }
         weight_vector_int  = _mm256_setr_epi32(temp_weight_static[0],temp_weight_static[1],temp_weight_static[2],temp_weight_static[3],temp_weight_static[4],temp_weight_static[5],temp_weight_static[6],temp_weight_static[7]);
       }
       else if (S>8 && k==1){
         for(int z=0; z<8; z++){
           temp_weight_static[z] = (z<(S-8))?temp_weight[s][m][l][8+z]:0;
           }
           weight_vector_int  = _mm256_setr_epi32(temp_weight_static[0],temp_weight_static[1],temp_weight_static[2],temp_weight_static[3],temp_weight_static[4],temp_weight_static[5],temp_weight_static[6],temp_weight_static[7]);
       }

       weight_vector_float = _mm256_cvtepi32_ps(weight_vector_int);
     
       for(int q=0; q<input_features->dim1; q++){
         for(int j=0; j<h; j++){
           for(int i=0; i<w; i++){

             input_vector_int = _mm256_setr_epi32(temp_in[q][m][j*Sy + l][i*Sx],temp_in[q][m][j*Sy + l][i*Sx+1],temp_in[q][m][j*Sy + l][i*Sx+2],temp_in[q][m][j*Sy + l][i*Sx+3],temp_in[q][m][j*Sy + l][i*Sx+4],temp_in[q][m][j*Sy + l][i*Sx+5],temp_in[q][m][j*Sy + l][i*Sx+6],temp_in[q][m][j*Sy + l][i*Sx+7]);  

             input_vector_float = _mm256_cvtepi32_ps(input_vector_int);
             product_vector_float = _mm256_mul_ps(input_vector_float,weight_vector_float);

             product_vector_int = _mm256_cvtps_epi32(product_vector_float);
             _mm256_storeu_si256((__m256i*)&temp[0],product_vector_int);

             temp_out[q][s][j][i] = temp_out[q][s][j][i] + temp[7] + temp[6] + temp[5] + temp[4] + temp[3] + temp[2] + temp[1] + temp[0];
           }
         }
       }
     }
    }
   }
  }

  end = clock();

  exec_time = double(end-start) / double(CLOCKS_PER_SEC);

  delete [] temp_in;

  output_features_pointer = &output_features;

  return output_features_pointer;
}

fmap* Convolution::conv2d_optimized(fmap* input_features)      //Convolution optimized and tested to work for Layer 2 of Alexnet
{
  fmap* output_features_pointer; fmap output_features;

  __m256i weight_vector_int, input_vector_int, product_vector;
  __m256  input_vector_float, product_vector_float, weight_vector_float;

  int temp[8];
  int t, index, output_index, n, c, h, w;
  int* temp_in_init;

  int T = 3; //TILE SIZE

  clock_t start,end;

  output_features.dim1 = n = input_features->dim1;
  output_features.dim2 = c = M;
  output_features.dim4 = w = (input_features->dim4-S+2*Px)/Sx + 1;
  output_features.dim3 = h = (input_features->dim3-R+2*Py)/Sy + 1;

  output_features.data = (DATA*) malloc(n * c * h * w * sizeof(DATA));
  temp_in_init = (int*) malloc(input_features->dim1 * input_features->dim2 * (input_features->dim3 + 2*Py) * (input_features->dim4 + 2*Px) * sizeof(int));

  DATA (*temp_out)[c][h][w]        = (DATA (*)[c][h][w])output_features.data;
  int (*temp_in_prel)[input_features->dim2][input_features->dim3][input_features->dim4]     = (DATA (*)[input_features->dim2][input_features->dim3][input_features->dim4])input_features->data;
  int (*temp_in) [input_features->dim2][input_features->dim3+(2*Py)][input_features->dim4+(2*Px)] = (int (*)[input_features->dim2][input_features->dim3+(2*Py)][input_features->dim4+(2*Px)])temp_in_init;
  int (*temp_weight)[C][R][S]      = (DATA (*)[C][R][S])weights;

  for(int l=0; l<input_features->dim1; l++)                                                                   //Data re-arrangement for padding
    for(int k=0; k<input_features->dim2; k++)
      for(int j=0; j<(input_features->dim3+2*Py); j++)
        for(int i=0; i<(input_features->dim4+2*Px); i++) 
          if(i<Px || ((input_features->dim4+2*Px)-i-1)<Px || j<Py || ((input_features->dim3+2*Py)-j-1)<Py)
            temp_in[l][k][j][i] = 0; 
          else
            temp_in[l][k][j][i] = temp_in_prel[l][k][j-Py][i-Px];

  delete [] input_features->data;

  start = clock();

  for(int q=0; q<n; q++){                     //Batch size loop
   for(int u=0; u<c; u++){                    //Number of output channels loop/number of filters loop
   for(int tiley=0; tiley<(h/T + 1); tiley++){
   for(int tilex=0; tilex<(w/T + 1); tilex++){
    for(int k=(tiley*T), ty=0; k<input_features->dim3 && ty<T; k++,ty++){                  //Number of outpupt colunmns loop

      for(int p=(tilex*T), tx=0; p<w && tx<T; p++,tx++){

          for(int m=0; m<C; m++){
               
            for(int i=0; i<8; i++){
              temp[i] = (i<S)?temp_in[q][m][k][p*Sx+i]:0;
            }
            input_vector_int   = _mm256_setr_epi32(temp[0],temp[1],temp[2],temp[3],temp[4],temp[5],temp[6],temp[7]);

            input_vector_float = _mm256_cvtepi32_ps(input_vector_int);
            
            if(k<R){
              t = k/Sy;

              index = k%R;
              for(int i=0;i<(t+1);i++){
                weight_vector_int   = _mm256_setr_epi32(temp_weight[u][m][index-(i*Sy)][0],temp_weight[u][m][index-(i*Sy)][1],temp_weight[u][m][index-(i*Sy)][2],temp_weight[u][m][index-(i*Sy)][3],temp_weight[u][m][index-(i*Sy)][4],temp_weight[u][m][index-(i*Sy)][5],temp_weight[u][m][index-(i*Sy)][6],temp_weight[u][m][index-(i*Sy)][7]);   //Load weights
                weight_vector_float = _mm256_cvtepi32_ps(weight_vector_int);                                     

                product_vector_float = _mm256_mul_ps(input_vector_float,weight_vector_float);                      //Multiply the vectors

                product_vector = _mm256_cvtps_epi32(product_vector_float);                                         //Convert float to integer
                _mm256_storeu_si256((__m256i*)&temp[0],product_vector);                                            //Store in memory
            
                temp_out[q][u][i][p] = temp_out[q][u][i][p] + temp[7] + temp[6] + temp[5] + temp[4] + temp[3] + temp[2] + temp[1] + temp[0];  //Local addition within vector
                
              }
            }
            else if(k<(input_features->dim3-R)){
              index = k - (k/Sy)*Sy;

              if(k<((t+1)*Sy) || k>(input_features->dim3-(t+1)*Sy)){   //Input vectors for which number of reuse is less due to bigger strides 
                for(int i=0; i<t; i++){
                  output_index = (k/Sy) - i;

                  weight_vector_int   = _mm256_setr_epi32(temp_weight[u][m][index+(i*Sy)][0],temp_weight[u][m][index+(i*Sy)][1],temp_weight[u][m][index+(i*Sy)][2],temp_weight[u][m][index+(i*Sy)][3],temp_weight[u][m][index+(i*Sy)][4],temp_weight[u][m][index+(i*Sy)][5],temp_weight[u][m][index+(i*Sy)][6],temp_weight[u][m][index+(i*Sy)][7]);
                  weight_vector_float = _mm256_cvtepi32_ps(weight_vector_int);

                  product_vector_float = _mm256_mul_ps(input_vector_float,weight_vector_float);

                  product_vector = _mm256_cvtps_epi32(product_vector_float);
                  _mm256_storeu_si256((__m256i*)&temp[0],product_vector);

                  temp_out[q][u][output_index][p] = temp_out[q][u][output_index][p] + temp[7] + temp[6] + temp[5] + temp[4] + temp[3] + temp[2] + temp[1] + temp[0];
                }
              }
              else{
                for(int i=0; i<(t+1); i++){
                  output_index = (k/Sy) - i;

                  weight_vector_int   = _mm256_setr_epi32(temp_weight[u][m][index+(i*Sy)][0],temp_weight[u][m][index+(i*Sy)][1],temp_weight[u][m][index+(i*Sy)][2],temp_weight[u][m][index+(i*Sy)][3],temp_weight[u][m][index+(i*Sy)][4],temp_weight[u][m][index+(i*Sy)][5],temp_weight[u][m][index+(i*Sy)][6],temp_weight[u][m][index+(i*Sy)][7]);
                  weight_vector_float = _mm256_cvtepi32_ps(weight_vector_int);

                  product_vector_float = _mm256_mul_ps(input_vector_float,weight_vector_float);

                  product_vector = _mm256_cvtps_epi32(product_vector_float);
                  _mm256_storeu_si256((__m256i*)&temp[0],product_vector);

                  temp_out[q][u][output_index][p] = temp_out[q][u][output_index][p] + temp[7] + temp[6] + temp[5] + temp[4] + temp[3] + temp[2] + temp[1] + temp[0];
                }
              }
            }
            else{
              index = R - (input_features->dim3 - k - 1) % R - 1;

              t = ((input_features->dim3-k)%R)/Sy;

              for(int i=0; i<(t+1); i++){
                output_index = (k/Sy) - i;

                weight_vector_int   = _mm256_setr_epi32(temp_weight[u][m][index+(i*Sy)][0],temp_weight[u][m][index+(i*Sy)][1],temp_weight[u][m][index+(i*Sy)][2],temp_weight[u][m][index+(i*Sy)][3],temp_weight[u][m][index+(i*Sy)][4],temp_weight[u][m][index+(i*Sy)][5],temp_weight[u][m][index+(i*Sy)][6],temp_weight[u][m][index+(i*Sy)][7]);
                weight_vector_float = _mm256_cvtepi32_ps(weight_vector_int);

                 product_vector_float = _mm256_mul_ps(input_vector_float,weight_vector_float);

                 product_vector = _mm256_cvtps_epi32(product_vector_float);
                 _mm256_storeu_si256((__m256i*)&temp[0],product_vector);

                 temp_out[q][u][output_index][p] = temp_out[q][u][output_index][p] + temp[7] + temp[6] + temp[5] + temp[4] + temp[3] + temp[2] + temp[1] + temp[0];
              }
            }
          }
       } }
      }
    }
   }
  }

  end = clock();
 
  double exec_time = double(end-start) / double(CLOCKS_PER_SEC);

  delete [] temp_in;

  output_features_pointer = &output_features;

  return output_features_pointer;

}

fmap* Linear::linear(fmap* input_features)
{
  fmap output_features; fmap* output_features_pointer;

  __m256i input_vector_int, weight_vector_int, output_vector_int;
  __m256  input_vector, weight_vector, product_vector, output_vector;

  DATA* temp_input; DATA* temp_output;

  int* start_pointer;
  int temp_vect_int[8], temp_input_vect[8], temp_weight_vect[8];
  int32_t* temp_vect; 

  clock_t start,end;

  output_features.dim4 = output_features.dim3 = output_features.dim1 = 1;
  output_features.dim2 = M;

  output_features.data = (DATA*) malloc(1 * M * 1 * 1 * sizeof(DATA));

  temp_input = input_features->data;
  temp_output = output_features.data;

  start = clock();

  for(int i=0; i < M; i++){                    //Iterate along each element of output

    start_pointer = (int*)temp_input;

    output_vector = _mm256_setr_ps(0,0,0,0,0,0,0,0);                           //Setting all registers to zero

    for(int j=0; j < (L/8); j++){              //Iterate in steps of 8 for each output

       if(L > 8){
         input_vector_int =  _mm256_setr_epi32(*(start_pointer + (j*8) + 0),*(start_pointer + (j*8) + 1), *(start_pointer + (j*8) + 2), *(start_pointer + (j*8) + 3), *(start_pointer + (j*8) + 4), *(start_pointer + (j*8) + 5), *(start_pointer + (j*8) + 6), *(start_pointer + (j*8) + 7));
         input_vector = _mm256_cvtepi32_ps(input_vector_int);

         weight_vector_int = _mm256_setr_epi32(*(weights + (i*L) + (j*8) + 0),*(weights + (i*L) + (j*8) + 1),*(weights + (i*L) + (j*8) + 2),*(weights + (i*L) + (j*8) + 3),*(weights + (i*L) + (j*8) + 4),*(weights + (i*L) + (j*8) + 5),*(weights + (i*L) + (j*8) + 6),*(weights + (i*L) + (j*8) + 7));
         weight_vector = _mm256_cvtepi32_ps(weight_vector_int);

         product_vector = _mm256_mul_ps(input_vector,weight_vector);

         output_vector = _mm256_add_ps(output_vector,product_vector);

       }

       if((L > (j+1)*8) && (j==(L/8)-1)){
         for(int q=0; q<8; q++){
           if(q < L-((j+1)*8)){  
             temp_input_vect[q]  = *(start_pointer + ((j+1)*8) + q); 
             temp_weight_vect[q] = *(weights + (i*L) + ((j+1)*8) + q);
           }
           else{
             temp_input_vect[q]  = 0;
             temp_weight_vect[q] = 0;
           }
        }

        input_vector_int = _mm256_setr_epi32(temp_input_vect[7],temp_input_vect[6],temp_input_vect[5],temp_input_vect[4],temp_input_vect[3],temp_input_vect[2],temp_input_vect[1],temp_input_vect[0]);
        weight_vector_int = _mm256_setr_epi32(temp_weight_vect[7],temp_weight_vect[6],temp_weight_vect[5],temp_weight_vect[4],temp_weight_vect[3],temp_weight_vect[2],temp_weight_vect[1],temp_weight_vect[0]);
                 
        input_vector  = _mm256_cvtepi32_ps(input_vector_int);
        weight_vector = _mm256_cvtepi32_ps(weight_vector_int);

        product_vector = _mm256_mul_ps(input_vector,weight_vector);
        output_vector = _mm256_add_ps(output_vector,product_vector);
      }
    }
    output_vector_int = _mm256_cvtps_epi32(output_vector);
           
    _mm256_storeu_si256((__m256i*)&temp_vect_int[0],output_vector_int);

    temp_vect = (int32_t *)temp_vect_int;

    temp_output[i] = temp_vect[0] + temp_vect[1] + temp_vect[2] + temp_vect[3] + temp_vect[4] + temp_vect[5] + temp_vect[6] + temp_vect[7];
  }

  end = clock();

  exec_time = double(end-start) / double(CLOCKS_PER_SEC);

  delete [] input_features->data; 

  output_features_pointer = &output_features; 

  return output_features_pointer;
}

fmap* Linear::linear_optimized(fmap* input_features)
{
  return NULL;
}

void relu(fmap* input_features)
{
  int* temp; int* base_addr;

  int n,c,h,w;

  int temp_input_vect[8], temp_vect_int[8];
  int horizontal_stride;
  int32_t* temp_vect;

  DATA* temp_input;

  __m256  input_vector, comparison_vector, output_vector;
  __m256i input_vector_int, output_vector_int;

  n = input_features->dim1;
  c = input_features->dim2;
  h = input_features->dim3;
  w = input_features->dim4;

  temp = (int*)input_features->data;
  temp_input = input_features->data;

  comparison_vector = _mm256_setr_ps(0,0,0,0,0,0,0,0);                                  //Set all zeros to registers

  for(int i=0; i < n; i++){
    for(int j=0; j < c; j++){
      for(int k=0; k<h; k++){

        horizontal_stride = 0;

        for(int l=0; l < ((w/8<1)?1:(w/8)); l++){

          base_addr = temp+(i*c*h*w)+(j*h*w)+(k*w)+(l*8);

          if(w >= 8){
            input_vector_int = _mm256_setr_epi32(*(base_addr+0), *(base_addr+1), *(base_addr+2), *(base_addr+3), *(base_addr+4), *(base_addr+5), *(base_addr+6), *(base_addr+7)); //Loaded the values to SIMD lanes
          
            input_vector = _mm256_cvtepi32_ps(input_vector_int);    //Converting to single precision float
            comparison_vector = _mm256_setr_ps(0,0,0,0,0,0,0,0); //Loaded a zero comparison vector

            output_vector = _mm256_max_ps(input_vector, comparison_vector);

            output_vector_int = _mm256_cvtps_epi32(output_vector);
           
            _mm256_storeu_si256((__m256i*)&temp_vect_int[0],output_vector_int);

            temp_vect = (int32_t *)temp_vect_int;

            for(int m=0; m<8 ; m++){ 
              temp_input[(i*c*h*w)+(j*h*w)+(k*w)+(l*8)+m] = temp_vect[m];
            }

          }

            if(((w > (l+1)*8) && (l==(w/8)-1)) || (w/8<1)){                       //Modulo of 8 values have to be dealt with here
              for(int q=0; q<8; q++){
                if(w>8){
                 if(q < w-((l+1)*8)){ 
                   temp_input_vect[q]  = *(base_addr + 8 + q); 
                   horizontal_stride = horizontal_stride + 1;
                 }
                 else
                   temp_input_vect[q]  = 0;

                 input_vector_int = _mm256_setr_epi32(temp_input_vect[0],temp_input_vect[1],temp_input_vect[2],temp_input_vect[3],temp_input_vect[4],temp_input_vect[5],temp_input_vect[6],temp_input_vect[7]);
                 
                 input_vector  = _mm256_cvtepi32_ps(input_vector_int);

                 output_vector = _mm256_max_ps(input_vector, comparison_vector);     //ReLu: Max(0,x)

                 output_vector_int = _mm256_cvtps_epi32(output_vector);
           
                 _mm256_storeu_si256((__m256i*)&temp_vect_int[0],output_vector_int);

                 temp_vect = (int32_t *)temp_vect_int;

                 for(int m=0; m<horizontal_stride ; m++){ 
                   temp_input[(i*c*h*w)+(j*h*w)+(k*w)+(l*8)+ 8 +m] = temp_vect[m];
                 }
               }  
               else{
                 if(q<w){
                   temp_input_vect[q]  = *(base_addr + 8 + q); 
                   horizontal_stride = horizontal_stride + 1;
                 }
                 else
                   temp_input_vect[q] = 0;

                 input_vector_int = _mm256_setr_epi32(temp_input_vect[0],temp_input_vect[1],temp_input_vect[2],temp_input_vect[3],temp_input_vect[4],temp_input_vect[5],temp_input_vect[6],temp_input_vect[7]);
                 
                 input_vector  = _mm256_cvtepi32_ps(input_vector_int);

                 output_vector = _mm256_max_ps(input_vector, comparison_vector);     //ReLu: Max(0,x)

                 output_vector_int = _mm256_cvtps_epi32(output_vector);
           
                 _mm256_storeu_si256((__m256i*)&temp_vect_int[0],output_vector_int);

                 temp_vect = (int32_t *)temp_vect_int;

                 for(int m=0; m<horizontal_stride ; m++){ 
                   temp_input[(i*c*h*w)+(j*h*w)+(k*w)+m] = temp_vect[m];
                 }
               }
             }
           }
        }
      } 
    }
  }
}

fmap* maxpool_2d(fmap* input_features, int R, int S, int Sx, int Sy)
{
 fmap maxpool_map; fmap* maxpool_map_pointer;
  DATA* temp_input; DATA* temp_output;

  int n,c,h,w;
  int temp_vect[8];
  int32_t* temp_vect_out;  

  int* start_pointer;

  __m256i input_vector_int, output_vector_int;
  __m256  input_vector, max_vector;

  n = input_features->dim1;
  c = input_features->dim2;
  h = input_features->dim3;
  w = input_features->dim4;

  maxpool_map.dim1 = input_features->dim1;
  maxpool_map.dim2 = input_features->dim2;
  maxpool_map.dim3 = (h-R)/Sy + 1;
  maxpool_map.dim4 = (w-S)/Sx + 1;

  maxpool_map.data = (DATA*) malloc(n * c * h * w * sizeof(DATA));

  temp_input  = input_features->data;
  temp_output = maxpool_map.data;

  
  for(int i=0; i<maxpool_map.dim1; i++){
    for(int j=0; j<maxpool_map.dim2; j++){
      for(int k=0; k<maxpool_map.dim3; k++){
        for(int l=0; l<maxpool_map.dim4; l++){

          start_pointer = (int*) temp_input;

          max_vector = _mm256_setr_ps(0,0,0,0,0,0,0,0);

          for(int m=0; m<R ; m++){
            input_vector_int = _mm256_setr_epi32(*(start_pointer),*(start_pointer+1),*(start_pointer+2),0,0,0,0,0);
            input_vector = _mm256_cvtepi32_ps(input_vector_int);

            start_pointer = start_pointer + w;

            max_vector = _mm256_max_ps(max_vector,input_vector);
          }

          output_vector_int = _mm256_cvtps_epi32(max_vector);
          _mm256_storeu_si256((__m256i*)&temp_vect[0],output_vector_int);

          temp_vect_out = (int32_t*)temp_vect;
          
          temp_output[(i*c*h*w)+(j*h*w)+(k*w)+l] = (temp_vect_out[2]>temp_vect_out[1])?((temp_vect_out[0]>temp_vect_out[2])?temp_vect_out[0]:temp_vect_out[2]):((temp_vect_out[0]>temp_vect_out[1])?temp_vect_out[0]:temp_vect_out[1]);          

          temp_input = input_features->data + i*c*h*w + j*h*w + k*w*Sy + ((l+1)*Sx);
        }
        
        temp_input = input_features->data + i*c*h*w + j*h*w + ((k+1)*Sy*w);
      }

      temp_input = input_features->data + i*c*h*w + (j+1)*h*w;
    }
    
    temp_input = input_features->data + (i+1)*c*h*w;
  }

  delete [] input_features->data;  

  maxpool_map_pointer = &maxpool_map;

  return maxpool_map_pointer;
}

AlexNet::AlexNet()
{
  conv_layers = (Convolution**) malloc(5 * sizeof(Convolution*));

  Convolution *conv;
  conv = new Convolution(96, 3, 11, 11, 4, 4, 0, 0);
  conv_layers[0] = conv;
  conv = new Convolution(256, 96, 5, 5, 1, 1, 2, 2);
  conv_layers[1] = conv;
  conv = new Convolution(384, 256, 3, 3, 1, 1, 1, 1);
  conv_layers[2] = conv;
  conv = new Convolution(384, 384, 3, 3, 1, 1, 1, 1);
  conv_layers[3] = conv;
  conv = new Convolution(256, 384, 3, 3, 1, 1, 1, 1);
  conv_layers[4] = conv;

  linear_layers = (Linear**) malloc(3 * sizeof(Linear*));

  Linear *linear;
  linear = new Linear(4096, 9216);
  linear_layers[0] = linear;
  linear = new Linear(4096, 4096);
  linear_layers[1] = linear;
  linear = new Linear(1000, 4096);
  linear_layers[2] = linear;
}

fmap* AlexNet::forward_pass(fmap* input_features)
{
  clock_t start, end;
  int32_t foo1, foo2, foo3, foo4; int32_t* foo_pointer;
  start = clock();

  fmap bridge;                                           //Need a local static memory as the stack crashes when relu calls a pre computed convolution member function

  fmap* temp = input_features;
  
  temp = conv_layers[0]->conv2d_IS(temp);

  foo1 = temp->dim1 ; foo2 = temp->dim2; foo3 = temp->dim3; foo4 = temp->dim4; foo_pointer = temp->data;
  bridge.dim1 = foo1; bridge.dim2 = foo2; bridge.dim3 = foo3; bridge.dim4 = foo4; bridge.data = foo_pointer;  

  relu(&bridge);
  temp = maxpool_2d(&bridge, 3, 3, 2, 2);

  foo1 = temp->dim1 ; foo2 = temp->dim2; foo3 = temp->dim3; foo4 = temp->dim4; foo_pointer = temp->data;
  bridge.dim1 = foo1; bridge.dim2 = foo2; bridge.dim3 = foo3; bridge.dim4 = foo4; bridge.data = foo_pointer;

  temp = conv_layers[1]->conv2d_IS(&bridge);

  foo1 = temp->dim1 ; foo2 = temp->dim2; foo3 = temp->dim3; foo4 = temp->dim4; foo_pointer = temp->data;
  bridge.dim1 = foo1; bridge.dim2 = foo2; bridge.dim3 = foo3; bridge.dim4 = foo4; bridge.data = foo_pointer;

  relu(&bridge);
  temp = maxpool_2d(&bridge, 3, 3, 2, 2);

  foo1 = temp->dim1 ; foo2 = temp->dim2; foo3 = temp->dim3; foo4 = temp->dim4; foo_pointer = temp->data;
  bridge.dim1 = foo1; bridge.dim2 = foo2; bridge.dim3 = foo3; bridge.dim4 = foo4; bridge.data = foo_pointer;

  temp = conv_layers[2]->conv2d_IS(&bridge);

  foo1 = temp->dim1 ; foo2 = temp->dim2; foo3 = temp->dim3; foo4 = temp->dim4; foo_pointer = temp->data;
  bridge.dim1 = foo1; bridge.dim2 = foo2; bridge.dim3 = foo3; bridge.dim4 = foo4; bridge.data = foo_pointer;

  relu(&bridge);
  temp = conv_layers[3]->conv2d_IS(&bridge);

  foo1 = temp->dim1 ; foo2 = temp->dim2; foo3 = temp->dim3; foo4 = temp->dim4; foo_pointer = temp->data;
  bridge.dim1 = foo1; bridge.dim2 = foo2; bridge.dim3 = foo3; bridge.dim4 = foo4; bridge.data = foo_pointer;

  relu(&bridge);
  temp = conv_layers[4]->conv2d_IS(&bridge);

  foo1 = temp->dim1 ; foo2 = temp->dim2; foo3 = temp->dim3; foo4 = temp->dim4; foo_pointer = temp->data;
  bridge.dim1 = foo1; bridge.dim2 = foo2; bridge.dim3 = foo3; bridge.dim4 = foo4; bridge.data = foo_pointer;

  relu(&bridge);
  temp = maxpool_2d(&bridge, 3, 3, 2, 2);

  foo1 = temp->dim1 ; foo2 = temp->dim2; foo3 = temp->dim3; foo4 = temp->dim4; foo_pointer = temp->data;
  bridge.data = foo_pointer;

  bridge.dim2 = foo2*foo3*foo4;;
  bridge.dim3 = bridge.dim4 = 1;
  bridge.dim1 = 1;

  temp = linear_layers[0]->linear(&bridge);

  foo1 = temp->dim1 ; foo2 = temp->dim2; foo3 = temp->dim3; foo4 = temp->dim4; foo_pointer = temp->data;
  bridge.dim1 = foo1; bridge.dim2 = foo2; bridge.dim3 = foo3; bridge.dim4 = foo4; bridge.data = foo_pointer;

  relu(&bridge);
  temp = linear_layers[1]->linear(&bridge);

  foo1 = temp->dim1 ; foo2 = temp->dim2; foo3 = temp->dim3; foo4 = temp->dim4; foo_pointer = temp->data;
  bridge.dim1 = foo1; bridge.dim2 = foo2; bridge.dim3 = foo3; bridge.dim4 = foo4; bridge.data = foo_pointer;

  relu(&bridge);
  temp = linear_layers[2]->linear(&bridge);

  foo1 = temp->dim1 ; foo2 = temp->dim2; foo3 = temp->dim3; foo4 = temp->dim4; foo_pointer = temp->data;
  bridge.dim1 = foo1; bridge.dim2 = foo2; bridge.dim3 = foo3; bridge.dim4 = foo4; bridge.data = foo_pointer;

  relu(&bridge);

  end = clock();

  exec_time = double(end-start) / double(CLOCKS_PER_SEC);
  return temp;
}
