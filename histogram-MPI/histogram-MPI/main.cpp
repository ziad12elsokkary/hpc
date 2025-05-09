#include <opencv2/opencv.hpp>
#include <vector>
#include <cmath>
#include <chrono>
#include <iostream>
#include <numeric>
#include <memory>
#include <mpi.h>

using namespace cv;
using namespace std;
using namespace std::chrono;

// ================== Global Variables ==================
int originalScale = 100;
int equalizedScale = 100;
const int maxScalePercentage = 500;
Mat displayedOriginalHist, displayedEqualizedHist;
Mat globalOutputImage;

struct ProcessingBuffers {
    vector<int> hist;
    vector<float> prob;
    vector<float> cumProb;
    vector<uchar> intensityMap;

    ProcessingBuffers() : hist(256, 0), prob(256, 0.0f),
        cumProb(256, 0.0f), intensityMap(256, 0) {}
};
ProcessingBuffers globalBuffers;
int globalTotalPixels = 0;
vector<double> globalTimings;

// ================== Visualization Functions ==================
Mat drawHistogram(const vector<int>& hist, const string& title, int totalPixels,
    float scaleFactor = 1.0f, const vector<double>& timings = {}) {
    const int histSize = hist.size();
    const int histWidth = 512, histHeight = 400;
    const int binWidth = cvRound(static_cast<double>(histWidth) / histSize);

    Mat histImg(histHeight, histWidth, CV_8UC3, Scalar(255, 255, 255));

    float maxVal = 0.0f;
    vector<float> normalized(histSize);
    for (int i = 0; i < histSize; i++) {
        normalized[i] = static_cast<float>(hist[i]) / totalPixels;
        maxVal = max(maxVal, normalized[i]);
    }

    float displayMax = maxVal * scaleFactor;

    for (int i = 0; i < histSize; i++) {
        float height = (normalized[i] / displayMax) * histHeight;
        height = min(height, static_cast<float>(histHeight));
        rectangle(histImg, Point(i * binWidth, histHeight - height),
            Point((i + 1) * binWidth, histHeight),
            Scalar(0, 0, 255), FILLED);
    }

    putText(histImg, title, Point(10, 30), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 0, 0), 2);
    stringstream scaleText;
    scaleText << "Scale: " << (scaleFactor * 100) << "%";
    putText(histImg, scaleText.str(), Point(histWidth - 150, 30),
        FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0, 0, 0), 1);

    if (!timings.empty()) {
        stringstream timingText;
        timingText << "Processing Time: " << timings[4] << "ms";
        putText(histImg, timingText.str(), Point(10, 60),
            FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0, 0, 0), 1);
    }

    return histImg;
}

void onOriginalTrackbar(int, void*) {
    float scaleFactor = originalScale / 100.0f;
    displayedOriginalHist = drawHistogram(globalBuffers.hist, "Original Histogram",
        globalTotalPixels, scaleFactor);
    imshow("Original Histogram", displayedOriginalHist);
}

void onEqualizedTrackbar(int, void*) {
    float scaleFactor = equalizedScale / 100.0f;
    vector<int> eqHist(256, 0);
    const uchar* outData = globalOutputImage.data;
    for (int i = 0; i < globalOutputImage.rows * globalOutputImage.cols; i++) {
        eqHist[outData[i]]++;
    }
    displayedEqualizedHist = drawHistogram(eqHist, "Equalized Histogram",
        globalTotalPixels, scaleFactor, globalTimings);
    imshow("Equalized Histogram", displayedEqualizedHist);
}

void visualizeResults(const Mat& input, const Mat& output,
    const ProcessingBuffers& buffers, int totalPixels,
    const vector<double>& timings) {
    globalBuffers = buffers;
    globalTotalPixels = totalPixels;
    globalTimings = timings;
    globalOutputImage = output.clone();

    namedWindow("Original Histogram", WINDOW_AUTOSIZE);
    namedWindow("Equalized Histogram", WINDOW_AUTOSIZE);

    createTrackbar("Zoom:", "Original Histogram", &originalScale,
        maxScalePercentage, onOriginalTrackbar);
    createTrackbar("Zoom:", "Equalized Histogram", &equalizedScale,
        maxScalePercentage, onEqualizedTrackbar);

    onOriginalTrackbar(0, 0);
    onEqualizedTrackbar(0, 0);

    Mat combined(input.rows, input.cols * 2, CV_8UC3);
    cvtColor(input, combined(Rect(0, 0, input.cols, input.rows)), COLOR_GRAY2BGR);
    cvtColor(output, combined(Rect(input.cols, 0, output.cols, output.rows)), COLOR_GRAY2BGR);
    imshow("Input vs Equalized", combined);
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    // Default paths for input and output images
    string inputPath = "F:\Senior2\Spring\hpc\histogram images\small.jpg";
    string outputPath = "equalized.jpg";
    
    // Process command-line arguments if provided
    if (rank == 0) {
        if (argc > 1) {
            inputPath = argv[1];
        }
        if (argc > 2) {
            outputPath = argv[2];
        }
        cout << "Input image path: " << inputPath << endl;
        cout << "Output image path: " << outputPath << endl;
    }

    Mat inputImage;
    ProcessingBuffers buffers;
    vector<double> timings;
    int totalPixels = 0;

    if (rank == 0) {
        inputImage = imread(inputPath, IMREAD_GRAYSCALE);
        if (inputImage.empty()) {
            cerr << "Error: Could not read image at path: " << inputPath << endl;
            MPI_Abort(MPI_COMM_WORLD, -1);
            return -1;
        }
        totalPixels = inputImage.rows * inputImage.cols;
    }

    int dimensions[2] = { 0, 0 };
    if (rank == 0) {
        dimensions[0] = inputImage.rows;
        dimensions[1] = inputImage.cols;
    }
    MPI_Bcast(dimensions, 2, MPI_INT, 0, MPI_COMM_WORLD);

    int rows = dimensions[0];
    int cols = dimensions[1];
    totalPixels = rows * cols;

    int chunk_size = totalPixels / size;
    int remainder = totalPixels % size;

    vector<int> recvcounts(size, chunk_size);
    vector<int> displs(size, 0);
    for (int i = 0; i < size; i++) {
        if (i < remainder) recvcounts[i]++;
        if (i > 0) displs[i] = displs[i - 1] + recvcounts[i - 1];
    }

    int local_size = recvcounts[rank];
    vector<uchar> local_pixels(local_size);

    MPI_Scatterv(rank == 0 ? inputImage.data : nullptr,
        recvcounts.data(), displs.data(), MPI_UNSIGNED_CHAR,
        local_pixels.data(), local_size, MPI_UNSIGNED_CHAR,
        0, MPI_COMM_WORLD);

    auto totalStart = high_resolution_clock::now();
    double stepTime;
    auto start = high_resolution_clock::now();

    // 1. Histogram
    vector<int> local_hist(256, 0);
    for (int i = 0; i < local_size; i++) {
        local_hist[local_pixels[i]]++;
    }

    buffers.hist.resize(256);
    MPI_Reduce(local_hist.data(), buffers.hist.data(), 256, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        stepTime = duration_cast<microseconds>(high_resolution_clock::now() - start).count() / 1000.0;
        timings.push_back(stepTime);
    }

    // 2. Probability (parallel)
 // 2. Probability (parallel via scatter/gather of histogram)
    buffers.prob.resize(256);
    vector<float> local_prob(256, 0.0f);

    int prob_chunk_size = 256 / size;
    int prob_remainder = 256 % size;

    vector<int> prob_counts(size, prob_chunk_size);
    vector<int> prob_displs(size, 0);
    for (int i = 0; i < size; i++) {
        if (i < prob_remainder) prob_counts[i]++;
        if (i > 0) prob_displs[i] = prob_displs[i - 1] + prob_counts[i - 1];
    }

    vector<int> local_hist_chunk(prob_counts[rank]);

    if (rank == 0) {
        // Already computed histogram on root; scatter chunks
        MPI_Scatterv(buffers.hist.data(), prob_counts.data(), prob_displs.data(),
            MPI_INT, local_hist_chunk.data(), prob_counts[rank], MPI_INT,
            0, MPI_COMM_WORLD);
    }
    else {
        // Receive histogram chunk
        MPI_Scatterv(nullptr, nullptr, nullptr,
            MPI_INT, local_hist_chunk.data(), prob_counts[rank], MPI_INT,
            0, MPI_COMM_WORLD);
    }

    start = high_resolution_clock::now();

    // Each process computes probability for its histogram chunk
    vector<float> local_prob_chunk(prob_counts[rank]);
    for (int i = 0; i < prob_counts[rank]; i++) {
        local_prob_chunk[i] = static_cast<float>(local_hist_chunk[i]) / totalPixels;
    }

    // Gather all probability chunks back to root
    MPI_Gatherv(local_prob_chunk.data(), prob_counts[rank], MPI_FLOAT,
        buffers.prob.data(), prob_counts.data(), prob_displs.data(),
        MPI_FLOAT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        stepTime = duration_cast<microseconds>(high_resolution_clock::now() - start).count() / 1000.0;
        timings.push_back(stepTime);
    }


    // 3. Cumulative (sequential)
    if (rank == 0) {
        start = high_resolution_clock::now();
        buffers.cumProb[0] = buffers.prob[0];
        for (int i = 1; i < 256; i++) {
            buffers.cumProb[i] = buffers.cumProb[i - 1] + buffers.prob[i];
        }
        stepTime = duration_cast<microseconds>(high_resolution_clock::now() - start).count() / 1000.0;
        timings.push_back(stepTime);
    }

    // Broadcast cumProb
    buffers.cumProb.resize(256);
    MPI_Bcast(buffers.cumProb.data(), 256, MPI_FLOAT, 0, MPI_COMM_WORLD);

    // 4. Intensity Map (parallel)
 // 4. Intensity Mapping (parallel)
    buffers.intensityMap.resize(256);

    int map_chunk_size = 256 / size;
    int map_remainder = 256 % size;

    vector<int> map_counts(size, map_chunk_size);
    vector<int> map_displs(size, 0);
    for (int i = 0; i < size; i++) {
        if (i < map_remainder) map_counts[i]++;
        if (i > 0) map_displs[i] = map_displs[i - 1] + map_counts[i - 1];
    }

    // Scatter cumProb to all processes
    vector<float> local_cumProb_chunk(map_counts[rank]);
    MPI_Scatterv(buffers.cumProb.data(), map_counts.data(), map_displs.data(), MPI_FLOAT,
        local_cumProb_chunk.data(), map_counts[rank], MPI_FLOAT,
        0, MPI_COMM_WORLD);

    start = high_resolution_clock::now();

    // Each process computes a chunk of the intensity map
    vector<uchar> local_intensity_chunk(map_counts[rank]);
    for (int i = 0; i < map_counts[rank]; i++) {
        local_intensity_chunk[i] = saturate_cast<uchar>(floor(local_cumProb_chunk[i] * 255));
    }

    // Gather intensity map chunks to root
    MPI_Gatherv(local_intensity_chunk.data(), map_counts[rank], MPI_UNSIGNED_CHAR,
        buffers.intensityMap.data(), map_counts.data(), map_displs.data(),
        MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        stepTime = duration_cast<microseconds>(high_resolution_clock::now() - start).count() / 1000.0;
        timings.push_back(stepTime);
    }

    // Broadcast full intensity map to all processes
    MPI_Bcast(buffers.intensityMap.data(), 256, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);


    // 5. Apply Mapping
    start = high_resolution_clock::now();
    for (int i = 0; i < local_size; i++) {
        local_pixels[i] = buffers.intensityMap[local_pixels[i]];
    }

    vector<uchar> output_pixels;
    if (rank == 0) {
        output_pixels.resize(totalPixels);
    }

    MPI_Gatherv(local_pixels.data(), local_size, MPI_UNSIGNED_CHAR,
        output_pixels.data(), recvcounts.data(), displs.data(),
        MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        stepTime = duration_cast<microseconds>(high_resolution_clock::now() - start).count() / 1000.0;
        timings.push_back(stepTime);

        Mat outputImage(rows, cols, CV_8UC1, output_pixels.data());
        double totalTime = duration_cast<microseconds>(high_resolution_clock::now() - totalStart).count() / 1000.0;

        cout << fixed << setprecision(3);
        cout << "\n=== Execution Times ===\n";
        cout << "1. Histogram:      " << timings[0] << " ms\n";
        cout << "2. Probabilities:  " << timings[1] << " ms\n";
        cout << "3. Cumulative:     " << timings[2] << " ms\n";
        cout << "4. Intensity Map:  " << timings[3] << " ms\n";
        cout << "5. Transformation: " << timings[4] << " ms\n";
        cout << "-----------------------\n";        cout << "Total Time:        " << timings[0]+timings[1]+timings[2]+timings[3]+timings[4] << " ms\n";
        cout << "Total solution Time:" << totalTime << " ms\n";
        cout << "Image Size:        " << cols << "x" << rows << endl;
        cout << "MPI Processes:     " << size << endl;

        imwrite(outputPath, outputImage);
        visualizeResults(inputImage, outputImage, buffers, totalPixels, timings);
        waitKey(0);
    }

    MPI_Finalize();
    return 0;
}
