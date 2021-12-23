#include <thread>
#include <iostream>

using namespace std;

void func() {
    for (int i = 0; i < 100; ++i)
        cout << i << "\n";
}

int main() {
    thread t1(func);
    thread t2(func);
    thread t3(func);
    thread t4(func);
    t1.join();
    t2.join();
    t3.join();
    t4.join();
    return 0;
}