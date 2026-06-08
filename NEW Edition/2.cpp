#include <Python.h>
#include <iostream>

int main() {
    Py_Initialize();

    // 执行 Python 代码打印 sys.path
    PyRun_SimpleString("import sys");
    PyRun_SimpleString("print('Python sys.path =', sys.path)");

    Py_Finalize();
    return 0;
}
