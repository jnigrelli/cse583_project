#include <vector>
#include <cstdlib>

using namespace std;

void int_test() {
    volatile int a = 1, b = 2, c = 3;
    for (long long i = 0; i < 200000000LL; i++) {
        a = b + c * i;
    }
}

void float_test() {
    volatile double a = 1.1, b = 2.2, c = 3.3;
    for (long long i = 0; i < 200000000LL; i++) {
        a = b + c * 1.00001;
    }
}

void mem_test() {
    int size = 1024 * 1024 * 64;
    vector<int> arr(size, 1);
    volatile int sum = 0;
    for (long long i = 0; i < 50000000LL; i++) {
        int idx = (i * 1000003) % size;
        sum += arr[idx];
    }
}

void branch_test() {
    volatile int a = 0;
    for (long long i = 0; i < 200000000LL; i++) {
        if ((i ^ 123456789) % 2 == 0) {
            a++;
        } else {
            a--;
        }
    }
}

int main(int argc, char** argv) {
    int type;
    if (argc < 2) {
        type = 0;
    } else {
        type = atoi(argv[1]);
    }

    if (type == 0) int_test();
    else if (type == 1) float_test();
    else if (type == 2) mem_test();
    else if (type == 3) branch_test();

    return 0;
}
