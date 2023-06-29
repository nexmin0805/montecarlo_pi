#include <iostream>
#include <vector>
#include <sys/time.h>

using namespace std;

void getGapTime(struct timeval* start_time, struct timeval* end_time, struct timeval* gap_time){
  gap_time->tv_sec = end_time->tv_sec - start_time->tv_sec;
  gap_time->tv_usec = end_time->tv_usec - start_time->tv_usec;
  if(gap_time->tv_usec < 0){
    gap_time->tv_usec = gap_time->tv_usec + 1000000;
    gap_time->tv_sec -= 1;
  }
}

float timevalToFloat(struct timeval* time){
    double val;
    val = time->tv_sec;
    val += (time->tv_usec * 0.000001);
    return val;
}
//커널 함수
__global__ void montecarlo_pi_kernel(float* device_randX, float* device_randY, int* device_blocks, int block_num, int dot) {

  __shared__ int s_data[1000];

  int index = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = blockDim.x * block_num;

  int circle = 0;  
  for (int i = index; i < dot; i+= stride) {    
    float xValue = device_randX[i];    
    float yValue = device_randY[i];
  //점이 원안에 찍히면 circle++
    if (xValue*xValue + yValue*yValue <= 1.0f) {
      circle++;    
    }  
  }
  
  s_data[threadIdx.x] = circle;

  __syncthreads();

  if (threadIdx.x == 0){    
    int total_block = 0;    
    for (int j = 0; j < blockDim.x; j++){      
      total_block += s_data[j];    
    }
    device_blocks[blockIdx.x] = total_block;  
  }
}

int dot = 100000000;

int main() { 
  //성능측정
  struct timeval htod_start, htod_end, htod_gap;
  struct timeval gpu_start, gpu_end, gpu_gap;
  struct timeval dtoh_start, dtoh_end, dtoh_gap;

  // 랜덤값 생성
  srand(time(NULL));   
  vector<float> host_randX(dot);    
  vector<float> host_randY(dot);

  
  //벡터 초기화
  for (int i = 0; i < host_randX.size(); ++i){        
    host_randX[i] = float(rand()) / RAND_MAX;        
    host_randY[i] = float(rand()) / RAND_MAX;    
  }
  

  size_t size = dot * sizeof(float);    
  float* device_randX;    
  float* device_randY;   

  

  //각 블록당 1000개의 스레드, 각 프로세스당 1000개의 점
  int threads_num = 1000;
  int block_num = dot / (1000 * threads_num);
  size_t block_size = block_num * sizeof(int);

  int* device_blocks;
  cudaMalloc(&device_blocks, block_size);
  cudaMalloc(&device_randX, size);  
  cudaMalloc(&device_randY, size);

  // 호스트에서 디바이스로 복사 
  gettimeofday(&htod_start, NULL);
  cudaMemcpy(device_randX, &host_randX.front(), size, cudaMemcpyHostToDevice);    
  cudaMemcpy(device_randY, &host_randY.front(), size, cudaMemcpyHostToDevice);
  gettimeofday(&htod_end, NULL);
  getGapTime(&htod_start, &htod_end, &htod_gap); 
  float f_htod_gap = timevalToFloat(&htod_gap);


  // 커널 호출 
  gettimeofday(&gpu_start, NULL);
  montecarlo_pi_kernel<<<block_num, threads_num>>>(device_randX, device_randY, device_blocks, block_num, dot);
  if ( cudaSuccess != cudaGetLastError() ){
    cout << "오류입니다.\n";
  }
  gettimeofday(&gpu_end, NULL);
  getGapTime(&gpu_start, &gpu_end, &gpu_gap);
  double f_gpu_gap = timevalToFloat(&gpu_gap);

  // 디바이스에서 호스트로 복사
  int* host_blocks = new int[block_num];
  gettimeofday(&dtoh_start, NULL);
  cudaMemcpy(host_blocks, device_blocks, block_size, cudaMemcpyDeviceToHost);
  gettimeofday(&dtoh_end, NULL);
  getGapTime(&htod_start, &dtoh_end, &dtoh_gap);
  float f_dtoh_gap = timevalToFloat(&dtoh_gap);

  //총 소요시간
  float total_gap = f_htod_gap + f_gpu_gap + f_dtoh_gap;

  int dot_in_circle = 0;
  for (int i = 0 ; i < block_num; i++) {
    dot_in_circle = dot_in_circle + host_blocks[i];
  }

  //메모리 해제
  cudaFree(device_randX);
  cudaFree(device_randY);
  cudaFree(device_blocks);

  // 파이 추정값 = (4*원 안에 찍힌 점의 개수) / (점의 총 개수)
  float pi = 4.0 * float(dot_in_circle) / dot;
  cout.precision(5);
  cout << "pi값: " << pi << endl;
  cout << "GPU 소요시간: " << total_gap <<","<< " htod time: " <<f_htod_gap<<","<< " GPU time: " << f_gpu_gap <<","<< " dtoh time" << f_dtoh_gap;

}
