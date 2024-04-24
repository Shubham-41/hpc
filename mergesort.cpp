#include<iostream>
using namespace std ;
#include<omp.h>
#include<vector>
#include<ctime>

void merge_sort(int arr[], int low, int high);
void merge_array(int arr[],int low, int mid, int high);

//sequential_merge_sort(arr,0,n);

void merge_sort(int arr[], int low, int high)
{
    if(low >= high)
        return ;

        int mid = (low+high)/2 ;

        #pragma omp parallel sections num_threads(2)

        # pragma omp  section
        merge_sort(arr,low,mid);

        # pragma omp  section
        merge_sort(arr,mid+1,high);

        merge_array(arr,low,mid,high);


}

void merge_array(int arr[],int low, int mid, int high)
{
    int left = low ;
    int right = mid+1 ;

    vector<int> temp ;

    while(left<=mid && right <= high)
    {
        if(arr[left]<= arr[right])
        {
            temp.push_back(arr[left]);
            left++;
        }
        else
        {
            temp.push_back(arr[right]);
            right++;
        }
    }

    while(left<=mid)
    {
        temp.push_back(arr[left]);
        left++;
    }

    while(right<=high)
    {
        temp.push_back(arr[right]);
        right++;
    }

    for(int i=low; i<= high;i++)
    {
        arr[i] = temp[i-low] ;
    }

}

int main()
{
    int n ;
    cout<<"Enter number of elements "<<endl;
    cin>>n;

    int arr[n] ;
    cout<<"Enter "<<n<<" Elements"<<endl;

    for(int i=0;i<n;i++)
    {
        cin>>arr[i];
    }

    clock_t start_time = clock();
    merge_sort(arr,0,n);
    clock_t end_time = clock();
    double parallel_time = double(end_time - start_time)/ CLOCKS_PER_SEC ;

    //start_time = clock();
    //sequential_merge_sort(arr,0,n);
    //end_time = clock();
    //double sequential_time = double(end_time-start_time)/ CLOCKS_PER_SEC ;



    for(int i=0;i<n;i++)
    {
        cout<<arr[i]<<" ";
    }

    //cout<<"Sequential Sort Time "<<sequential_time<<" seconds"<<endl;

    cout<<"Parallel Sort Time "<<parallel_time<<"  seconds"<<endl;




}
