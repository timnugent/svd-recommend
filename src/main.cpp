// SVD-like Matrix Factorization by Tim Nugent 2015

#include <iostream>
#include <sys/stat.h>
#include "svd_collabfilt.h"

using namespace std;

// Helper functions to check if file exists
bool exists (const std::string& name) {
    struct stat buffer;   
    return (stat(name.c_str(), &buffer) == 0); 
}

int main(int argc, const char* argv[]){

    string infile = "";
    string outfile = "predictions.tsv";
    unsigned int top_predictions = 25;
    unsigned int threads = 4;
    unsigned int k = 240;

    if(argc < 2){
        cout << argv[0] << " [-t <threads> -p <predictions/user> -o <output-tsv>] <input-tsv>" << endl;
        return(0);
    }else if(argc == 2){    
        infile = string(argv[1]);
    }else{
        //cout << "# called with:       ";
        for(int i = 0; i < argc; i++){
            //cout << argv[i] << " ";
            if(string(argv[i]) == "-i" && i < argc-1){
                infile = string(argv[i+1]);
            }
            if(string(argv[i]) == "-o" && i < argc-1){
                outfile = string(argv[i+1]);
            }
            if(string(argv[i]) == "-p" && i < argc-1){
                top_predictions = atoi(argv[i+1]);
            }
            if(string(argv[i]) == "-t" && i < argc-1){
                threads = atoi(argv[i+1]);
            }
            if(string(argv[i]) == "-h"){
                cout << argv[0] << " [-t <threads> -p <predictions/user> -o <output-tsv>] <input-tsv>" << endl;
                return(0);
            }
        }
    }
    if(!exists(infile)){
        cout << infile << " doesn't exist!" << endl;
        return(0);
    }

    cout << "Input file:      " << infile << endl;
    cout << "Output file:     " << outfile << endl;
    cout << "Predictions:     " << top_predictions << endl;
    cout << "Latent features: " << k << endl;
    cout << "Threads:         " << threads << endl;
    
    SVD* S = new SVD();
    S->set_features(k);   
    S->set_min_epochs(100);
    S->set_max_epochs(10000); 
    S->set_stochastic(true);   
    S->set_alpha(0.0001);
    S->set_lambda(0.02);
    S->set_round(false);
    S->set_minpred(1.0);
    S->set_maxpred(5.0);    
    S->set_verbose(true);
    S->set_threads(8);
    S->read_training_tsv(infile.c_str()); 
    S->set_init(0.1);
    S->set_lambda(0.1);
    S->set_min_epochs(100);
    S->set_features(k);  
    S->set_verbose(false);   
    S->factorize2();
    S->set_verbose(true);
    S->write_pq_matrices("data/P_demo.mat","data/Q_demo.mat");
    S->write_top_predictions(outfile.c_str(),25);   

    delete S;
    return(0);
    
}
