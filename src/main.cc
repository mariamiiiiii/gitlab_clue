#include <stdlib.h>
#include <unistd.h>

#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <regex>
#include <string>

#include "CLUEAlgo.h"

#if defined(USE_ALPAKA)
#include "CLUEAlgoAlpaka.h"
#else
#include "CLUEAlgoGPU.h"
#endif

#ifdef ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLED
#include "tbb/global_control.h"
#endif

#define NLAYERS 100

using namespace std;

void exclude_stats_outliers(std::vector<float> &v) {
  if (v.size() == 1)
    return;
  float mean = std::accumulate(v.begin(), v.end(), 0.0) / v.size();
  float sum_sq_diff =
      std::accumulate(v.begin(), v.end(), 0.0, [mean](float acc, float x) {
        return acc + (x - mean) * (x - mean);
      });
  float stddev = std::sqrt(sum_sq_diff / (v.size() - 1));
  std::cout << "Sigma cut outliers: " << stddev << std::endl;
  float z_score_threshold = 3.0;
  v.erase(std::remove_if(v.begin(), v.end(),
                         [mean, stddev, z_score_threshold](float x) {
                           float z_score = std::abs(x - mean) / stddev;
                           return z_score > z_score_threshold;
                         }),
          v.end());
}

pair<float, float> stats(const std::vector<float> &v) {
  float m = std::accumulate(v.begin(), v.end(), 0.0) / v.size();
  float sum = std::accumulate(v.begin(), v.end(), 0.0, [m](float acc, float x) {
    return acc + (x - m) * (x - m);
  });
  auto den = v.size() > 1 ? (v.size() - 1) : v.size();
  return {m, std::sqrt(sum / den)};
}

void printTimingReport(std::vector<float> &vals, int repeats,
                       std::vector<std::pair<std::string, float>> &timings,
                       const std::string label = "SUMMARY ") {
  int precision = 2;
  float mean = 0.f;
  float sigma = 0.f;
  exclude_stats_outliers(vals);
  tie(mean, sigma) = stats(vals);
  std::cout << label << " 1 outliers(" << repeats << "/" << vals.size() << ") "
            << std::fixed << std::setprecision(precision) << mean << " +/- "
            << sigma << " [ms]" << std::endl;
  exclude_stats_outliers(vals);
  tie(mean, sigma) = stats(vals);
  std::cout << label << " 2 outliers(" << repeats << "/" << vals.size() << ") "
            << std::fixed << std::setprecision(precision) << mean << " +/- "
            << sigma << " [ms]" << std::endl;

  if (label == "SUMMARY WorkDivByPoints submission copy_to_device times:") {
    timings.emplace_back("SubmissionCopyToDevice", mean);
  }     
  else if (label == "SUMMARY WorkDivByPoints execution copy_to_device times:") {
    timings.emplace_back("ExecutionCopyToDevice", mean);
  }
  else if (label == "SUMMARY WorkDivByPoints submission make_clusters times:") {
    timings.emplace_back("SubmissionMakeClusters", mean);
  }
  else if (label == "SUMMARY WorkDivByPoints execution make_clusters times:") {
    timings.emplace_back("ExecutionMakeClusters", mean);
  }
  else if (label == "SUMMARY WorkDivByPoints submission copy_to_host times:") {
    timings.emplace_back("SubmissionCopyToHost", mean);
  }
  else if (label == "SUMMARY WorkDivByPoints execution copy_to_host times:") {
    timings.emplace_back("ExecutionCopyToHost", mean);
  }
}

void reserveInputData(std::vector<float> &x, std::vector<float> &y,
                std::vector<int> &layer, std::vector<float> &weight, 
                int capacity) {
  x.reserve(capacity);
  y.reserve(capacity);
  layer.reserve(capacity);
  weight.reserve(capacity);
}

void readDataFromFile(const std::string &inputFileName, std::vector<float> &x,
                      std::vector<float> &y, std::vector<int> &layer,
                      std::vector<float> &weight, int capacity, int &size) {

  int i = 0;

  for (int l = 0; l < NLAYERS; l++) {
    // open csv file
    std::ifstream iFile(inputFileName);
    std::string value = "";
    // Iterate through each line and split the content using delimeter
    while (getline(iFile, value, ',')) {
      if(i < capacity) {
        x.push_back(std::stof(value));
        getline(iFile, value, ',');
        y.push_back(std::stof(value));
        getline(iFile, value, ',');
        layer.push_back(std::stoi(value) + l);
        getline(iFile, value);
        weight.push_back(std::stof(value));
      }
      else{
        std::cerr << "Error: Capacity exceeded (" << capacity << "). Exiting..." << std::endl;
        exit(EXIT_FAILURE);
      }
      i++;
    }
    iFile.close();
  }
  size = i;
}

void hostRegisterData(std::vector<float> &x, std::vector<float> &y, 
                      std::vector<int> &layer, std::vector<float> &weight) {
  cudaHostRegister(x.data(), x.size() * sizeof(float), cudaHostRegisterPortable | cudaHostRegisterMapped);
  cudaHostRegister(y.data(), y.size() * sizeof(float), cudaHostRegisterPortable | cudaHostRegisterMapped);
  cudaHostRegister(layer.data(), layer.size() * sizeof(int), cudaHostRegisterPortable | cudaHostRegisterMapped);
  cudaHostRegister(weight.data(), weight.size() * sizeof(float), cudaHostRegisterPortable | cudaHostRegisterMapped);
}

void freeInputData(std::vector<float> &x, std::vector<float> &y,
                std::vector<int> &layer, std::vector<float> &weight) {
  auto x_ = std::move(x);
  auto y_ = std::move(y);
  auto layer_ = std::move(layer);
  auto weight_ = std::move(weight);
}

void freeOutputData(std::vector<float> &rho, std::vector<float> &delta,
                std::vector<unsigned int> &nearestHigher, std::vector<int> &clusterIndex, 
                std::vector<uint8_t> &isSeed) {
  auto rho_ = std::move(rho);
  auto delta_ = std::move(delta);
  auto nearestHigher_ = std::move(nearestHigher);
  auto clusterIndex_ = std::move(clusterIndex);
  auto isSeed_ = std::move(isSeed);
}

std::string create_outputfileName(const std::string &inputFileName,
                                  const float dc, const float rhoc,
                                  const float outlierDeltaFactor) {
  //  C++20
  //  auto suffix = std::format("_{:.2f}_{:.2f}_{:.2f}.csv", dc, rhoc,
  //  outlierDeltaFactor);
  char suffix[100];
  snprintf(suffix, 100, "_dc_%.2f_rho_%.2f_outl_%.2f.csv", dc, rhoc,
           outlierDeltaFactor);

  std::string tmpFileName;
  std::regex regexp("input");
  std::regex_replace(back_inserter(tmpFileName), inputFileName.begin(),
                     inputFileName.end(), regexp, "output");

  std::string outputFileName;
  std::regex regexp2(".csv");
  std::regex_replace(back_inserter(outputFileName), tmpFileName.begin(),
                     tmpFileName.end(), regexp2, suffix);

  return outputFileName;
}

void mainRun(const std::string &inputFileName,
             const std::string &outputFileName, const float dc,
             const float rhoc, const float outlierDeltaFactor,
             const bool use_accelerator, const int repeats,
             const bool verbose, char* argv[]) {

  cudaFree(nullptr);

  //////////////////////////////
  // read toy data from csv file
  //////////////////////////////

  unsigned int capacity = 1000000;
  int size;

  std::cout << "Start to load input points" << std::endl;

  std::vector<std::pair<std::string, float>> timings;

  std::vector<float> x;
  std::vector<float> y;
  std::vector<int> layer;
  std::vector<float> weight;

  std::vector<float> rho;
  std::vector<float> delta;
  std::vector<unsigned int> nearestHigher; 
  std::vector<int> clusterIndex;
  std::vector<uint8_t> isSeed;

  std::cout << "Finished loading input points" << std::endl;

  vector<float> vals;
  vector<float> vals2;
  vector<float> vals3;
  vector<float> vals4;
  vector<float> vals5;
  vector<float> vals6;

  auto begin = std::chrono::high_resolution_clock::now();

  reserveInputData(x, y, layer, weight, capacity);

  auto end = std::chrono::high_resolution_clock::now();
  
  float time_reserve = std::chrono::duration<float>(end - begin).count();

  begin = std::chrono::high_resolution_clock::now();

  readDataFromFile(inputFileName, x, y, layer, weight, capacity, size);

  end = std::chrono::high_resolution_clock::now();
  
  float time_read = std::chrono::duration<float>(end - begin).count();

  timings.emplace_back("readDataFromFile", time_read * 1000);

  begin = std::chrono::high_resolution_clock::now();

  hostRegisterData(x, y, layer, weight);

  end = std::chrono::high_resolution_clock::now();
  
  float time_host_register = std::chrono::duration<float>(end - begin).count();

  float time_allocate_input_cpu = time_reserve + time_host_register;

  //////////////////////////////
  // run CLUE algorithm
  //////////////////////////////
  std::cout << "Start to run CLUE algorithm" << std::endl;
  if (use_accelerator) {
#if !defined(USE_ALPAKA)
    std::cout << "Native CUDA Backend selected" << std::endl;
    CLUEAlgoGPU<TilesConstants, NLAYERS> clueAlgo(dc, rhoc, outlierDeltaFactor,
                                                  verbose);

    vals.clear();
    vals2.clear();
    vals3.clear();
    vals4.clear();
    vals5.clear();
    vals6.clear();

    begin = std::chrono::high_resolution_clock::now();

    clueAlgo.init_input_data();

    clueAlgo.Sync();

    end = std::chrono::high_resolution_clock::now();
  
    float time_allocate_input_gpu = std::chrono::duration<float>(end - begin).count();
     
    begin = std::chrono::high_resolution_clock::now();

    clueAlgo.resizeOutputContainers(size);

    end = std::chrono::high_resolution_clock::now();
  
    float time_allocate_output_cpu = std::chrono::duration<float>(end - begin).count();
    
    begin = std::chrono::high_resolution_clock::now();

    clueAlgo.init_output_data();

    clueAlgo.Sync();

    end = std::chrono::high_resolution_clock::now();
  
    float time_allocate_output_gpu = std::chrono::duration<float>(end - begin).count();

    timings.emplace_back("allocateInputData", (time_allocate_input_cpu + time_allocate_input_gpu) * 1000);

    timings.emplace_back("allocateOutputData", (time_allocate_output_cpu + time_allocate_output_gpu) * 1000);

    for (unsigned r = 0; r < repeats; r++) {
      if (!clueAlgo.setPoints(x.size(), &x[0], &y[0], &layer[0], &weight[0]))
        exit(EXIT_FAILURE);
      clueAlgo.Sync();
      auto start = std::chrono::high_resolution_clock::now();
      clueAlgo.copy_todevice();
      auto finish = std::chrono::high_resolution_clock::now();
      clueAlgo.Sync();
      auto finish2 = std::chrono::high_resolution_clock::now();
      std::chrono::duration<float> submit_copy_to_device = finish - start;
      std::chrono::duration<float> execute_copy_to_device = finish2 - start;
      start = std::chrono::high_resolution_clock::now();
      clueAlgo.makeClusters(); //without size, i can get point_.n
      finish = std::chrono::high_resolution_clock::now();
      clueAlgo.Sync();
      finish2 = std::chrono::high_resolution_clock::now();
      std::chrono::duration<float> submit_make_clusters = finish - start;
      std::chrono::duration<float> execute_make_clusters = finish2 - start;
      start = std::chrono::high_resolution_clock::now();
      clueAlgo.copy_tohost();
      finish = std::chrono::high_resolution_clock::now();
      clueAlgo.Sync();
      finish2 = std::chrono::high_resolution_clock::now();
      std::chrono::duration<float> submit_copy_to_host = finish - start;
      std::chrono::duration<float> execute_copy_to_host = finish2 - start;
      std::cout << "Iteration " << r;
      std::cout << " | Submission time copy_to_device: " << submit_copy_to_device.count() * 1000 << " ms\n";
      std::cout << " | Execution time copy_to_device: " << execute_copy_to_device.count() * 1000 << " ms\n";
      std::cout << " | Submission time make_clusters: " << submit_make_clusters.count() * 1000 << " ms\n";
      std::cout << " | Execution time make_clusters: " << execute_make_clusters.count() * 1000 << " ms\n";
      std::cout << " | Submission time copy_to_host: " << submit_copy_to_host.count() * 1000 << " ms\n";
      std::cout << " | Execution time copy_to_host: " << execute_copy_to_host.count() * 1000 << " ms\n";
      // Skip first event
      if (r != 0 or repeats == 1) {
        vals.push_back(submit_copy_to_device.count() * 1000);
        vals2.push_back(execute_copy_to_device.count() * 1000);
        vals3.push_back(submit_make_clusters.count() * 1000);
        vals4.push_back(execute_make_clusters.count() * 1000);
        vals5.push_back(submit_copy_to_host.count() * 1000);
        vals6.push_back(execute_copy_to_host.count() * 1000);
      }
    }

    printTimingReport(vals, repeats, timings, "SUMMARY WorkDivByPoints submission copy_to_device times:");
    printTimingReport(vals2, repeats, timings, "SUMMARY WorkDivByPoints execution copy_to_device times:");
    printTimingReport(vals3, repeats, timings, "SUMMARY WorkDivByPoints submission make_clusters times:");
    printTimingReport(vals4, repeats, timings, "SUMMARY WorkDivByPoints execution make_clusters times:");
    printTimingReport(vals5, repeats, timings, "SUMMARY WorkDivByPoints submission copy_to_host times:");
    printTimingReport(vals6, repeats, timings, "SUMMARY WorkDivByPoints execution copy_to_host times:");
    
    auto begin = std::chrono::high_resolution_clock::now();

    // output result to outputFileName. -1 means all points.
    clueAlgo.verboseResults(outputFileName, -1);

    auto end = std::chrono::high_resolution_clock::now();

    float time_write = std::chrono::duration<float>(end - begin).count();

    timings.emplace_back("writeDataToFile", time_write * 1000);

    begin = std::chrono::high_resolution_clock::now();
    
    clueAlgo.free_input_data();

    clueAlgo.Sync();
    
    end = std::chrono::high_resolution_clock::now();

    float time_free_input_gpu = std::chrono::duration<float>(end - begin).count();

    begin = std::chrono::high_resolution_clock::now();
    
    clueAlgo.free_output_data();

    clueAlgo.Sync();

    end = std::chrono::high_resolution_clock::now();

    float time_free_output_gpu = std::chrono::duration<float>(end - begin).count();

    begin = std::chrono::high_resolution_clock::now();
    
    freeInputData(x, y, layer, weight);

    end = std::chrono::high_resolution_clock::now();

    float time_free_input_cpu = std::chrono::duration<float>(end - begin).count();
    
    begin = std::chrono::high_resolution_clock::now();
    
    freeOutputData(rho, delta, nearestHigher, clusterIndex, isSeed);

    end = std::chrono::high_resolution_clock::now();

    float time_free_output_cpu = std::chrono::duration<float>(end - begin).count();

    timings.emplace_back("freeInputData", (time_free_input_cpu + time_free_input_gpu) * 1000); 

    timings.emplace_back("freeOutputData", (time_free_output_cpu + time_free_output_gpu) * 1000); 

#endif
  } else {
    std::cout << "Native CPU(serial) Backend selected" << std::endl;
    CLUEAlgo<TilesConstants, NLAYERS> clueAlgo(dc, rhoc, outlierDeltaFactor,
                                               verbose);
    vals.clear();
    for (int r = 0; r < repeats; r++) {
      if (!clueAlgo.setPoints(x.size(), &x[0], &y[0], &layer[0], &weight[0]))
        exit(EXIT_FAILURE);
      // measure excution time of makeClusters
      auto start = std::chrono::high_resolution_clock::now();
      clueAlgo.makeClusters();
      auto finish = std::chrono::high_resolution_clock::now();
      std::chrono::duration<float> elapsed = finish - start;
      std::cout << "Elapsed time: " << elapsed.count() * 1000 << " ms\n";
      // Skip first event
      if (r != 0 or repeats == 1) {
        vals.push_back(elapsed.count() * 1000);
      }
    }

    printTimingReport(vals, repeats, timings, "SUMMARY Native CPU:");
    // output result to outputFileName. -1 means all points.
    if (verbose)
      clueAlgo.verboseResults(outputFileName, -1);
  }

  std::string run_number = argv[13];
  std::string filename = "Results/results_classic" + run_number + ".csv";

  std::ofstream results(filename);
  if (!results.is_open()) {
    std::cerr << "Failed to open file.\n";
    return;
  }

  results << "Operation,Time\n";
  for (const auto& entry : timings) {
      results << entry.first << "," << entry.second << "\n";
  }

  results.close();

  std::cout << "Finished running CLUE algorithm" << std::endl;
} // end of testRun()

int main(int argc, char *argv[]) {
  //////////////////////////////
  // MARK -- set algorithm parameters
  //////////////////////////////

  extern char *optarg;

  bool use_accelerator = false;
  bool verbose = false;
  float dc = 20.f, rhoc = 80.f, outlierDeltaFactor = 2.f;
  int repeats = 10;
  int TBBNumberOfThread = 1;
  int opt;
  std::string inputFileName;

  while ((opt = getopt(argc, argv, "i:d:r:o:e:t:uv")) != -1) {
    switch (opt) {
    case 'i': /* input filename */
      inputFileName = string(optarg);
      break;
    case 'd': /* delta_c */
      dc = stof(string(optarg));
      break;
    case 'r': /* critical density */
      rhoc = stof(string(optarg));
      break;
    case 'o': /* outlier factor */
      outlierDeltaFactor = stof(string(optarg));
      break;
    case 'e': /* number of repeated session(s) a the selected input file */
      repeats = stoi(string(optarg));
      break;
    case 't': /* number of TBB threads */
      TBBNumberOfThread = stoi(string(optarg));
      std::cout << "Using " << TBBNumberOfThread;
      std::cout << " TBB Threads" << std::endl;
      break;
    case 'u': /* Use accelerator */
      use_accelerator = true;
      break;
    case 'v': /* Verbose output */
      verbose = true;
      break;
    default:
      std::cout << "bin/main -i [fileName] -d [dc] -r [rhoc] -o "
                   "[outlierDeltaFactor] -e [repeats] -t "
                   "[NumTBBThreads] -u -v"
                << std::endl;
      exit(EXIT_FAILURE);
    }
  }

#ifdef ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLED
  if (verbose) {
    std::cout << "Setting up " << TBBNumberOfThread << " TBB Threads"
              << std::endl;
  }
  tbb::global_control init(tbb::global_control::max_allowed_parallelism,
                           TBBNumberOfThread);
#endif

  //////////////////////////////
  // MARK -- set input and output files
  //////////////////////////////
  std::cout << "Input file: " << inputFileName << std::endl;

  std::string outputFileName =
      create_outputfileName(inputFileName, dc, rhoc, outlierDeltaFactor);
  std::cout << "Output file: " << outputFileName << std::endl;

  //////////////////////////////
  // MARK -- test run
  //////////////////////////////
  mainRun(inputFileName, outputFileName, dc, rhoc, outlierDeltaFactor,
          use_accelerator, repeats, verbose, argv);

  return 0;
}
