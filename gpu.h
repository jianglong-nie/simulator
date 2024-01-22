#ifndef GPU_H
#define GPU_H

#include <algorithm>
#include <vector>
#include "flow.h"

class GPU {
public:
    int rank;
    float dataSize;
    bool isFinished;
    std::vector<Flow> flows; // Vector of Flow objects

    // 有关滑动窗口的变量
    float len; // len = dataSize;
    float Cnvl;
    float Cnet;
    float Wnvl;
    float Wnet;
    float left = -1;
    float right = -1;
    float timeChunkZero = FLOAT_MAX; // Time of the zero chunk
    float timeChunkNow = 0; // Current time
    float timeChunkLast = timeChunkZero; // Last time a chunk was sent
    float alpha; // Alpha value for adjusting window size
    float delta; // Delta value for adjusting window size

    // GPU构造函数
    GPU(int rank, float dataSize, bool isFinished);

    void init(int rank, float dataSize, bool isFinished);

    //对滑动窗口的初始化
    void initWindow(float len, float Cnvl, float Cnet, float Wnvl, float Wnet, float alpha, float delta);

    bool rankFinish(std::string protocol);

    void sendChunk(std::string protocol);

    void computing(float unitTime);

    void communication(float unitTime);

    void isWorkFinished();

    void step(float unitTime);

    void control(float unitTime);
};

#endif // GPU_H