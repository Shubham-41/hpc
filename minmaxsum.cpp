//Assignment:Implement Min, Max, Sum and Average operations using Parallel Reduction

#include<bits/stdc++.h>

using namespace std ;

void min_reduction(int arr[], int n)
{
    int min_val = INT_MAX ;
    #pragma omp parallel for reduction(min:min_val)

    for(int i=0;i<n;i++)
    {
        if(arr[i]<min_val)
        {
            min_val = arr[i] ;
        }
    }

    cout<<"Minimum Value is "<<min_val<<endl;
}

void max_reduction(int arr[],int n)
{
    int max_val = arr[0] ;

    #pragma omp parallel for reduction(max,max_val)
        for(int i=0;i<n;i++)
        {
            if(arr[i]>max_val)
            {
                max_val = arr[i] ;
            }
        }

    cout<<"Maximum Value is "<<max_val<<endl;
}

void sum_reduction(int arr[],int n)
{
    int sum = 0;

    #pragma omp parallel for reduction (+:sum)
    {
        for(int i = 0 ; i < n ; i++ )
        {
            sum = sum + arr[i] ;
        }
    }

    cout<<"Sum of array elements is "<<sum<<endl;
}

void average_reduction(int arr[],int n)
{
    int sum = 0 ;

    #pragma omp parallel for reduction(+:sum)
    {
        for(int i=0;i<n;i++)
        {
            sum += arr[i] ;
        }
    }

    double average = (double) sum / n ;

    cout<<"Average of array elements is "<<average<<endl;
}


int main()
{
    int n ;
    cout<<"Enter size of an array";
    cin>>n;

    int arr[n] ;
    cout<<"Enter "<<n<<" Elements"<<endl;

    for(int i=0;i<n;i++)
    {
        cin>>arr[i];
    }


    min_reduction(arr,n);

    max_reduction(arr,n);

    sum_reduction(arr,n);

    average_reduction(arr,n);
}
