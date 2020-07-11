//
//  main.c
//  TSN_part_a
//
//  Created by Admin on 7/6/20.
//  Copyright © 2020 Mohamed Abdelhamid. All rights reserved.
//

#include <stdio.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <OpenCL/opencl.h>
#define DATA_SIZE (16)

//kernel code
const char *KernelSource = "\n" \
"__kernel void packet_switch(                                                  \n" \
"   __global float* input_headers,                                              \n" \
"   __global float* output_q1,                                     \n" \
"   __global float* output_q2,                                     \n" \
"   __global float* output_q3,                                     \n" \
"   const unsigned int count)                                           \n" \
"{                                                                      \n" \
"   int i = get_global_id(0);                                           \n" \
"   if(i < count)                                                       \n" \
"       output[i] = input[i] * input[i];                                \n" \
"}                                                                      \n" \
"\n";
//Host Code

int main(int argc, const char * argv[]) {
   int err;                            // error code returned from api calls
     
   int data_switch_headers[DATA_SIZE];              //data headers set given to device
   int data_retreived_queue1[DATA_SIZE];           // results returned from device
   int data_retreived_queue2[DATA_SIZE];           // results returned from device
   int data_retreived_queue3[DATA_SIZE];           // results returned from device
    
   unsigned int correct;               // number of correct results returned

    size_t global[3] = {256,1,1};                                    // global domain  size for our calculation
   size_t local[3] = {16,1,1};                       // local domain size for our calculation

   cl_device_id device_id;             // compute device id
   cl_context context;                 // compute context
   cl_command_queue commands;          // compute command queue
   cl_program program;                 // compute program
   cl_kernel kernel;                   // compute kernel
   
   cl_mem input_headers;                       // device memory used for the input array
   cl_mem output_q1;                      // device memory used for the output array
   cl_mem output_q2;                      // device memory used for the output array
   cl_mem output_q3;                      // device memory used for the output array
    
    srand(time(NULL));
    
    for(int i = 0; i < DATA_SIZE; i++){
        data_switch_headers[i] = (rand() % 3) + 1;
        printf("header number %d count %d \n", data_switch_headers[i], i);
    }
     
    int count_headers = 16;
   err = clGetDeviceIDs(NULL,CL_DEVICE_TYPE_GPU, 1, &device_id, NULL);
      if (err != CL_SUCCESS)
      {
          printf("Error: Failed to create a device group!\n");
          return EXIT_FAILURE;
      }
   context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
     if (!context)
     {
         printf("Error: Failed to create a compute context!\n");
         return EXIT_FAILURE;
     }
    commands = clCreateCommandQueue(context, device_id, 0, &err);
       if (!commands)
       {
           printf("Error: Failed to create a command commands!\n");
           return EXIT_FAILURE;
       }
    
    program = clCreateProgramWithSource(context, 1, (const char **) & KernelSource, NULL, &err);
       if (!program)
       {
           printf("Error: Failed to create compute program!\n");
           return EXIT_FAILURE;
       }
   err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
   if (err != CL_SUCCESS)
   {
       size_t len;
       char buffer[2048];

       printf("Error: Failed to build program executable!\n");
       clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
       printf("%s\n", buffer);
       exit(1);
   }
    kernel = clCreateKernel(program, "packet_switch", &err);
       if (!kernel || err != CL_SUCCESS)
       {
           printf("Error: Failed to create compute kernel!\n");
           exit(1);
       }
    input_headers = clCreateBuffer(context,  CL_MEM_READ_ONLY,  sizeof(int) * count_headers, NULL, NULL);
    output_q1 = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(int) * count_headers, NULL, NULL);
    output_q2 = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(int) * count_headers, NULL, NULL);
    output_q3 = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(int) * count_headers, NULL, NULL);
     
    err = clEnqueueWriteBuffer(commands, input_headers, CL_TRUE, 0, sizeof(int) * count_headers, data_switch_headers, 0, NULL, NULL);
       if (err != CL_SUCCESS)
       {
           printf("Error: Failed to write to source array!\n");
           exit(1);
       }
    err = 0;
    err  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &input_headers);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &output_q1);
    err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &output_q2);
    err |= clSetKernelArg(kernel, 3, sizeof(cl_mem), &output_q3);
    err |= clSetKernelArg(kernel, 4, sizeof(unsigned int), &count_headers);
        if (err != CL_SUCCESS)
        {
            printf("Error: Failed to set kernel arguments! %d\n", err);
            exit(1);
        }
        
    err = clGetKernelWorkGroupInfo(kernel, device_id, CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, sizeof(local), &local, NULL);
    if (err != CL_SUCCESS)
      {
          printf("Error: Failed to retrieve kernel work group info! %d\n", err);
          exit(1);
      }
    err = clEnqueueNDRangeKernel(commands, kernel, 1, NULL, global, local, 0, NULL, NULL);
       if (err)
       {
           printf("Error: Failed to execute kernel!\n");
           return EXIT_FAILURE;
       }
    clFinish(commands);
     err = clEnqueueReadBuffer( commands, output_q1, CL_TRUE, 0, sizeof(int) * count_headers, data_retreived_queue1, 0, NULL, NULL );
       if (err != CL_SUCCESS)
       {
           printf("Error: Failed to read output queue1 ! %d\n", err);
           exit(1);
       }
    err = clEnqueueReadBuffer( commands, output_q2, CL_TRUE, 0, sizeof(int) * count_headers, data_retreived_queue2, 0, NULL, NULL );
        if (err != CL_SUCCESS)
        {
            printf("Error: Failed to read output array! %d\n", err);
            exit(1);
        }
    err = clEnqueueReadBuffer( commands, output_q3, CL_TRUE, 0, sizeof(int) * count_headers, data_retreived_queue3, 0, NULL, NULL );
          if (err != CL_SUCCESS)
          {
              printf("Error: Failed to read output queue 3! %d\n", err);
              exit(1);
          }
    
    
    //Releasing the kernel
    clReleaseMemObject(input_headers);
    clReleaseMemObject(output_q1);
    clReleaseMemObject(output_q2);
    clReleaseMemObject(output_q3);
    clReleaseProgram(program);
    clReleaseKernel(kernel);
    clReleaseCommandQueue(commands);
    clReleaseContext(context);
    return 0;
}
