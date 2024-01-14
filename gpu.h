#ifndef GPU_H
#define GPU_H

#include <vector>
#include "flow.h"

class GPU {
public:
    int rank;
    int dataSize;
    bool isFinished;
    std::vector<Flow> flows; // Vector of Flow objects

    // GPU构造函数
    GPU(int rank, int dataSize, bool isFinished);

    void init(int rank, int dataSize, bool isFinished);

    void computing(int unitTime);

    void communication(int unitTime);

    void checkFlows();

    void step(int unitTime);

    void control();
};

#endif // GPU_H