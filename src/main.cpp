// SVD-like Matrix Factorization by Tim Nugent 2015

#include <iostream>
#include "svd_collabfilt.h"

using namespace std;

// Helper functions to sort a map by value
template<typename A, typename B> pair<B,A> flip_pair(const pair<A,B> &p){
    return pair<B,A>(p.second, p.first);
}
template<typename A, typename B> multimap<B,A> flip_map(const map<A,B> &src){
    multimap<B,A> dst;
    transform(src.begin(), src.end(), inserter(dst, dst.begin()), flip_pair<A,B>);
    return dst;
}

int main(){

    // Demo run with 5 latent features
    unsigned int rank = 5;
    SVD* S = new SVD();
    S->set_features(rank);
    S->set_max_epochs(5000);
    S->set_min_epochs(10000);
    S->set_srand();
    S->read_training_tsv("data/demo.tsv");
    S->factorize();  
    S->write_pq_matrices("data/P_demo.mat","data/Q_demo.mat");
    S->calc_rmse();
    cout << endl << "Original partial matrix:" << endl;
    cout << S->get_x() << endl;
    cout << endl << "Low-rank (" << rank << ") approximation:" << endl;
    cout << S->get_p()*S->get_q() << endl << endl;

    // Optimise the number of latent features
    cout << "Searching for optimal number of latent features (optimizes features one at a time):" << endl;        
    unsigned int min_k = 1, max_k = 20;
    map<unsigned int, double> results;
    S->set_verbose(false);
    for(unsigned int k = min_k; k <= max_k; k++){
        S->set_features(k);     
        S->factorize();
        cout << "Features: " << k << "\t";
        results[k] = S->calc_rmse();
    }
    cout << endl;
    multimap<double, unsigned int> m = flip_map(results);
    cout << "Features\tRMSE" << endl;
    for(auto it = m.begin(); it != m.end(); it++){
        printf("%i\t\t%2.6f\n",it->second,it->first);
    }
    cout << endl;

    // Optimise the number of latent features
    cout << "Searching for optimal number of latent features: (optimizes features simultaneously)" << endl;  
    results.clear();
    S->set_verbose(false);
    for(unsigned int k = min_k; k <= max_k; k++){
        S->set_features(k);     
        S->factorize2();
        cout << "Features: " << k << "\t";
        results[k] = S->calc_rmse();
    }
    cout << endl;  

    m = flip_map(results);
    cout << "Features\tRMSE" << endl;
    for(auto it = m.begin(); it != m.end(); it++){
        printf("%i\t\t%2.6f\n",it->second,it->first);
    }
    cout << endl;

    delete S;
    return(0);
}
