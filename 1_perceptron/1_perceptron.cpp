#include<iostream>
#include<vector>
using namespace std;
class RPerceptron{

public:
        float learning_rate, threshold, iterations; // necessary vars
        vector< float > weights; //keeping it 1D for now     
        
        RPerceptron( float lr, int iters = 100){
            learning_rate = lr;
            iterations = iters;
            threshold = 0;
        }
        
    
        int activation(float weighted_sum){
            if (weighted_sum >= threshold)
            return 1;
            else
            return 0;
        }
        
        // training starts
        void train(vector < vector < int > > X, vector <int> y)
        {
            int num_features = X[0].size(); //this gets the number of features/columns
            weights.assign(num_features, 0); // initializing weights
            threshold = 0;
            // intermediate states 
            float weighted_sum = 0;
            float y_pred = 0;
            float error = 0;
            for(int iter = 0; iter<iterations; ++iter) //epoch
            {   
                for(int j=0; j<X.size(); ++j) //rows
                {
                 weighted_sum = 0;
                 y_pred =0;
                 error = 0;

                 for(int k = 0 ; k<X[0].size(); ++k) // columns
                 {
                 weighted_sum+=weights[k] * X[j][k]; // each element(weighted sum)
                 }
                 y_pred = activation(weighted_sum); // as the name says
                 error = y[j] - y_pred;

                // weight update accordin to the paper
                for( int k = 0; k<num_features; ++k){
                    weights[k] = weights[k] + (learning_rate * error * X[j][k]);

                // updating threhsold accoring to the paper
                threshold = threshold - (learning_rate * error); 

                }

                }
                cout<<"Iteration: "<<iter<<", erorr: "<<error<<", threshold= "<<threshold<<endl;
            }




        } 
        vector <int> predict(vector < vector < int > > X ){
            vector < int > predictions;
            for( int i = 0; i<X.size(); ++i)
            { float weighted_sum = 0;
                for ( int j = 0; j < X[i].size(); ++j)
                {
                    weighted_sum += (weights[j] * X[i][j]);
                }

                predictions.push_back(activation(weighted_sum));
            }
            return predictions;
        } 


};;

int main(){

// COnsider it as input to AND gate
vector < vector < int > > X = {
    {0,0},
    {0,1},
    {1,0},
    {1,1}
}; 

// output from AND gate
vector <int> y = {0,0,0,1};

//  predictions;
RPerceptron perceptron(0.1, 10);
perceptron.train(X, y);

vector <int> predictions = perceptron.predict(X);

for (int i =0; i<predictions.size(); ++i)
cout<<predictions[i]<< " ";
}