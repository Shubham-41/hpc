// Design and implement Parallel Depth First Search based on existing algorithms using OpenMP. Use a  undirected graph for DFS.
// Practise 01

#include<iostream>
using namespace std ;
#include<vector>
#include<omp.h>


void parallel_dfs(vector<vector<int>>& graph , vector<int>& visited, int start , int num_nodes)
{
    if(visited[start]==0)
    {

        cout<<" "<<start;

        visited[start] = 1 ;

        // Parallelize the loop using OpenMP

        # pragma omp parallel for
        for(int v = 1; v <= num_nodes ; v++ )
        {

            if(graph[start][v]==1 && visited[v]==0)
            {

                parallel_dfs(graph,visited,v,num_nodes);
            }
        }
    }
}

int main()
{
    int num_nodes , num_edges ;
    cout<<"Enter number of nodes and number of edges :"<<endl;
    cin>>num_nodes>>num_edges;

    vector<vector<int>> graph(num_nodes+1, vector<int>(num_nodes+1,0));

    vector<int> visited(num_nodes+1,0) ;

    int u , v ;

    for(int i=0;i<num_edges;i++)
    {
        cout<<"Enter (u:v) ";

        cin>>u>>v;

        graph[u][v] = 1 ;

        graph[v][u] = 1 ;
    }

    int start;
    cout<<"Enter starting vertex:"<<endl;
    cin>>start;

    cout<<"Parallel DFS :"<<" Starting From vertex "<<start<<endl;

    parallel_dfs(graph,visited,start,num_edges);

}
