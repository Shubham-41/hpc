// Write a program to implement Parallel Bubble Sort. Use existing algorithms and measure the performance of sequential and parallel algorithms.

#include<iostream>
using namespace std ;
#include<vector>
#include<omp.h>
#include<ctime>
#include<bits/stdc++.h>

void sequential_bubblesort(int arr[], int n)
{
    for(int i=0;i< n-1 ;i++)
    {
        for(int j = i ; j < n-i-1 ; j++ )
        {
            if(arr[j]>arr[j+1])
            {
                int temp = arr[j] ;
                arr[j] = arr[j+1];
                arr[j+1] = temp ;
            }
        }
    }
}

void parallel_bubblesort(int arr[], int n)
{
    for(int i=0;i<n-1;i++)
    {
        #pragma omp parallel for

        for(int j = i ; j < n-i-1 ; j++ )
        {
            if(arr[j] > arr[j+1])
            {
                int temp = arr[j] ;
                arr[j] = arr[j+1];
                arr[j+1] = temp ;
            }
        }
    }

}

int main()
{
   /*
    int n ;
    cout<<"Enter number of elements"<<endl;
    cin>>n;

    int arr[n];

    for(int i=0;i<n;i++)
    {
        cin>>arr[i]
        parallel_Arr[i] = arr[i];
    }

    */

    const int size = 10000 ;

    int arr[size] ;
    int parallel_Arr[size];

    // Initialize array with random values

    srand(time(NULL));
    for(int i =0; i < size;i++)
    {
        arr[i] = rand()%1000;
        parallel_Arr[i] = arr[i];
    }

    // Sequential Bubble Sort
    clock_t start_time = clock();
    sequential_bubblesort(arr,size);
    clock_t end_time = clock();
    double sequential_time = double(end_time-start_time)/CLOCKS_PER_SEC ;

    // Parallel Bubble Sort
    start_time = clock();
    parallel_bubblesort(parallel_Arr,size);
    end_time = clock();
    double parallel_bubblesort = double(end_time-start_time)/CLOCKS_PER_SEC ;

    // Printing Output Results

    cout<<"Sequential Bubble Sort "<<sequential_time<<" seconds"<<endl;

    cout<<"Parallel Bubble Sort "<<parallel_bubblesort<<" seconds"<<endl;

    // Verify both arrays are sorted

    for(int i=0;i<size;i++)
    {
        if(arr[i] != parallel_Arr[i])
        {
            cout<<"Sorting Mismatch at index "<<i<<" "<<endl;
            break;
        }
    }


}
