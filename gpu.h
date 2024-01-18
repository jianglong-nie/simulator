#ifndef GPU_H
#define GPU_H

#include <vector>
#include "flow.h"

class GPU {
public:
    int rank;
    float dataSize;
    bool isFinished;
    std::vector<Flow> flows; // Vector of Flow objects

    // GPU构造函数
    GPU(int rank, float dataSize, bool isFinished);

    void init(int rank, float dataSize, bool isFinished);

    void computing(float unitTime);

    void communication(float unitTime);

    void checkFlows();

    void step(float unitTime);

    void control();
};

#endif // GPU_H