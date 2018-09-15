#include "DataStructures/Stack.h"
// #include "Priority/GPR.h"
// #include "Priority/LocalEdgePriority.h"
// #include "Utilities/ScopeTime.h"
// #include "ImportsExports/ImportExportRagPriority.h"
#include <fstream>
#include <sstream>
#include <cassert>
#include <iostream>
#include <memory>
// #include <json/json.h>
// #include <json/value.h>

#include "Utilities/h5read.h"
#include "Utilities/h5write.h"

#include <ctime>
#include <cmath>
#include <cstring>

using std::cerr; using std::cout; using std::endl;
using std::ifstream;
using std::string;
using std::stringstream;
using namespace NeuroProof;
using std::vector;

typedef std::set<unsigned int> uint_array;


bool found_edge(std::map <unsigned int, uint_array>& prag_neighbors,
                unsigned int pnode1,
                unsigned int pnbr1)
{
    std::map <unsigned int, uint_array>::iterator rni;
    rni = prag_neighbors.find(pnode1);
    if (rni!=prag_neighbors.end()){
        uint_array& nbr_list = (rni->second);
        if (nbr_list.find(pnbr1)!=nbr_list.end())
            return true;
    }
    return false;
}

void add_edge(std::map <unsigned int, uint_array>& prag_neighbors,
                unsigned int pnode1,
                unsigned int pnbr1){
    std::map <unsigned int, uint_array>::iterator rni;
    rni = prag_neighbors.find(pnode1);
    if (rni!=prag_neighbors.end()){
        uint_array& nbr_list = (rni->second);
        nbr_list.insert(pnbr1);
    }
    else{
        std::set <unsigned int> arr1;
        arr1.insert(pnbr1);
        prag_neighbors.insert(std::pair<unsigned int, uint_array>(pnode1, arr1));
    }
    
}

int main(int argc, char** argv) 
{
    int          i, j, k;


    cout<< "Reading data ..." <<endl;



    if (argc<1){
	printf("format: synaptic_partner -watershed watershed_h5_file  dataset \n");
	return 0;
    }	

    int argc_itr=1;	
    string watershed_filename="";
    string watershed_dataset_name="";		
    string prediction_filename="";
    string prediction_dataset_name="";		
    string output_filename;
    string output_dataset_name;		
    string groundtruth_filename="";
    string groundtruth_dataset_name="";		
    string classifier_filename;

    string output_filename_nomito;

    	
    double threshold = 0.2;	
    int zbuffer = 0;		
    int xybuffer = 0;		
    bool merge_mito = true;
    bool merge_mito_by_chull = false;
    bool read_off_rwts = false;
    double mito_thd=0.3;
    size_t min_region_sz=100;
    while(argc_itr<argc){
	if (!(strcmp(argv[argc_itr],"-watershed"))){
	    watershed_filename = argv[++argc_itr];
	    watershed_dataset_name = argv[++argc_itr];
	}
    if (!(strcmp(argv[argc_itr],"-zbuffer"))){
        zbuffer = atoi(argv[++argc_itr]);
    }

    if (!(strcmp(argv[argc_itr],"-xybuffer"))){
        xybuffer = atoi(argv[++argc_itr]);
    }

        ++argc_itr;
    } 	
    	

    time_t start, end;
    time(&start);	

    H5Read watershed(watershed_filename.c_str(),watershed_dataset_name.c_str());	
    unsigned int* watershed_data=NULL;	
    watershed.readData(&watershed_data);	
    int depth =	 watershed.dim()[0];
    int height = watershed.dim()[1];
    int width =	 watershed.dim()[2];

    unsigned int curr_spot=0;
    
    
    std::map <unsigned int, uint_array> rag_neighbors;
    std::map <unsigned int, uint_array>::iterator rni;
   
    std::cout<<"zbuffer = "<<zbuffer<<" xybuffer = "<<xybuffer<<std::endl;
 
    for (unsigned int z = 1+zbuffer; z < (depth-1)-zbuffer; ++z) {
        int z_spot = z * (height*width);
        for (unsigned int y = 1+xybuffer; y < (height-1)-xybuffer; ++y) {
            int y_spot = y * width;
            for (unsigned int x = 1+xybuffer; x < (width-1)-xybuffer; ++x) {
                curr_spot = x + y_spot + z_spot;
                unsigned int spot0 = watershed_data[curr_spot];
                unsigned int spot1 = watershed_data[curr_spot-1];
                unsigned int spot2 = watershed_data[curr_spot+1];
                unsigned int spot3 = watershed_data[curr_spot-width];
                unsigned int spot4 = watershed_data[curr_spot+width];
                unsigned int spot5 = watershed_data[curr_spot-(height*width)];
                unsigned int spot6 = watershed_data[curr_spot+(height*width)];

                if (!spot0)
                    continue;

		// *C* point on the border with other superpixel
                if (spot1 && (spot0 != spot1)) {
                    if (!found_edge(rag_neighbors,spot0, spot1)){
                        //std::cout<< "("<<spot0<<", "<<spot1<< ")"<<std::endl;
                        add_edge(rag_neighbors, spot0, spot1);
                    }
                }
                if (spot2 && (spot0 != spot2) ) {
                    if (!found_edge(rag_neighbors,spot0, spot2)){
                        //std::cout<< "("<<spot0<<", "<<spot1<< ")"<<std::endl;
                        add_edge(rag_neighbors, spot0, spot2);
                    }
                }
                if (spot3 && (spot0 != spot3) ) {
                    if (!found_edge(rag_neighbors,spot0, spot3)){
                        //std::cout<< "("<<spot0<<", "<<spot1<< ")"<<std::endl;
                        add_edge(rag_neighbors, spot0, spot3);
                    }
                }
                if (spot4 && (spot0 != spot4) ) {
                    if (!found_edge(rag_neighbors,spot0, spot4)){
                        //std::cout<< "("<<spot0<<", "<<spot1<< ")"<<std::endl;
                        add_edge(rag_neighbors, spot0, spot4);
                    }
                }
                if (spot5 && (spot0 != spot5) ) {
                    if (!found_edge(rag_neighbors,spot0, spot5)){
                        //std::cout<< "("<<spot0<<", "<<spot1<< ")"<<std::endl;
                        add_edge(rag_neighbors, spot0, spot5);
                    }
                }
                if (spot6 && (spot0 != spot6) ) {
                    if (!found_edge(rag_neighbors,spot0, spot6)){
                        //std::cout<< "("<<spot0<<", "<<spot1<< ")"<<std::endl;
                        add_edge(rag_neighbors, spot0, spot6);
                    }
                }
                

            }
        }
    } 
    
    FILE* fid_edge = fopen("rag_edges.txt","wt");
    rni = rag_neighbors.begin();
    for(; rni!=rag_neighbors.end(); rni++){
        fprintf(fid_edge,"%u",rni->first);
        uint_array& nbr_list = (rni->second);
        uint_array::iterator nit = nbr_list.begin();
        for (; nit!=nbr_list.end(); nit++)
            fprintf(fid_edge,"  %u",*nit);
        fprintf(fid_edge,"\n");
    }
    fclose(fid_edge);
    

    if (watershed_data)  	
	delete[] watershed_data;
	

    return 0;
}
