//Design and implement Parallel Breadth First Search based on existing algorithms using OpenMP. Use  an undirected graph for BFS.
// Practise 01

#include<iostream>
using namespace std;
#include<vector>
#include<queue>
#include<omp.h>

void BFS(vector<vector<int>>& graph, int start , int nodes)
{
    vector<int> visited(nodes+1,0) ;

    queue <int> q ;

    visited[start] = 1 ;

    q.push(start);

    cout<<start<<" ";



    while(!q.empty())
    {

        int u = q.front();
        q.pop();

        for(int v =1 ; v<=nodes; v++)
        {
            if(graph[u][v]==1 && visited[v]==0 )
            {
                cout<<v<<" ";

                q.push(v);

                visited[v] = 1 ;

            }

        }
    }

}


void parallel_BFS(vector<vector<int>>& graph, int start, int num_nodes)
{
    vector<int> visited(num_nodes+1,0) ;

    queue<int> q ;

    q.push(start);

    visited[start] = 1 ;

    cout<<start<<" ";

    while(!q.empty())
    {
        #pragma omp parallel
        {
            int u;

            #pragma omp critical
            {

                u  = q.front();

                q.pop();

            }

            # pragma omp for

            for(int v = 1 ; v <= num_nodes ; v++)
            {
                if(graph[u][v]==1 && visited[v]==0)
                {

                    #pragma omp critical
                    {
                        visited[v] = 1 ;

                        cout<<v<<" ";

                        q.push(v);

                    }
                }

            }
        }

    }


}
int main()
{
    int num_nodes , num_edges ;
    cout<<"Enter number of nodes and edges"<<endl;
    cin>>num_nodes>>num_edges;

    vector<vector<int>> graph(num_nodes + 1, vector<int>(num_nodes + 1, 0));

    cout<<"Enter edges (u,v):"<<endl;

    for(int i=0 ; i < num_edges;i++ )
    {
        int u,v ;
        cin>>u>>v;
        graph[u][v] = 1 ;
        graph[v][u] = 1 ;
    }

    int start_node ;
    cout<<"Enter the starting node"<<endl;
    cin>>start_node;

    cout<<"BFS traversal starting from node"<< start_node << ": ";
    parallel_BFS(graph,start_node,num_nodes);
    cout<<endl;

    return 0;
}
